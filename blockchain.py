"""On-chain anchoring client.

The contract is a *notary*, not a custodian: it records that a digest existed at
a time, and that nobody has altered that record since. It says nothing about
whether the file behind the digest is genuine, and nothing about who really
submitted it - see contracts/EvidenceRegistry.sol for the full statement of what
this does and does not prove.

Two rules shape this module:

1. **Anchoring must never block the request.** A block takes roughly 12 seconds
   on Sepolia, and the network can be down. Registration succeeds regardless;
   the custody entry records ``anchor_status='failed'`` and a retry can come
   later. A notary outage must not stop someone saving evidence.

2. **Sign locally.** The private key never leaves this process: transactions are
   signed here and only the raw transaction is sent to the node.

Configuration comes from the environment:

    SEPOLIA_RPC_URL              e.g. https://eth-sepolia.g.alchemy.com/v2/<key>
    REGISTRY_CONTRACT_ADDRESS    0x... (printed by scripts/deploy_contract.py)
    ANCHOR_PRIVATE_KEY           0x... throwaway key, never a personal wallet
    CHAIN_EXPLORER_URL           optional, defaults to Sepolia Etherscan
"""

import json
import logging
import os
from pathlib import Path

from web3 import Web3

logger = logging.getLogger("forenscreen.blockchain")

BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BACKEND_DIR / "contracts" / "EvidenceRegistry.json"

DEFAULT_EXPLORER = "https://sepolia.etherscan.io"
DEFAULT_CHAIN_ID = 11155111  # Sepolia

# A mapping write plus an event; measured at ~46k on the deployed contract.
GAS_LIMIT = 200_000

_web3 = None
_contract = None
_private_key = None
_chain_id = None
_explorer = None


class NotConfigured(RuntimeError):
    """Raised when anchoring is attempted without chain configuration."""


def load_abi():
    if not ARTIFACT_PATH.exists():
        raise NotConfigured(
            f"{ARTIFACT_PATH.name} is missing. Run scripts/compile_contract.py first."
        )
    with open(ARTIFACT_PATH) as handle:
        return json.load(handle)["abi"]


def use(web3_instance, contract_address, private_key=None, explorer_url=None, chain_id=None):
    """Point the client at a specific chain.

    Used by tests and by anything running against a local node. Production uses
    the environment via `_connect()`.
    """
    global _web3, _contract, _private_key, _chain_id, _explorer
    _web3 = web3_instance
    _contract = _web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=load_abi())
    _private_key = private_key
    _chain_id = chain_id if chain_id is not None else _web3.eth.chain_id
    _explorer = explorer_url or DEFAULT_EXPLORER
    return _contract


def _connect():
    global _web3, _contract, _private_key, _chain_id, _explorer

    if _contract is not None:
        return _contract

    rpc_url = os.getenv("SEPOLIA_RPC_URL")
    address = os.getenv("REGISTRY_CONTRACT_ADDRESS")
    if not rpc_url or not address:
        raise NotConfigured(
            "Anchoring is not configured: set SEPOLIA_RPC_URL and "
            "REGISTRY_CONTRACT_ADDRESS (see .env.example)."
        )

    _web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 20}))
    _contract = _web3.eth.contract(
        address=Web3.to_checksum_address(address), abi=load_abi()
    )
    _private_key = os.getenv("ANCHOR_PRIVATE_KEY")
    _chain_id = int(os.getenv("CHAIN_ID", DEFAULT_CHAIN_ID))
    _explorer = os.getenv("CHAIN_EXPLORER_URL", DEFAULT_EXPLORER)
    return _contract


def is_configured() -> bool:
    return bool(os.getenv("SEPOLIA_RPC_URL") and os.getenv("REGISTRY_CONTRACT_ADDRESS"))


def to_digest(sha256_hex: str) -> bytes:
    """A SHA-256 hex string as the 32 raw bytes the contract expects."""
    digest = bytes.fromhex(sha256_hex.strip().lower())
    if len(digest) != 32:
        raise ValueError(f"Expected a 32-byte SHA-256 digest, got {len(digest)} bytes.")
    return digest


def explorer_url(tx_hash: str) -> str:
    return f"{_explorer or DEFAULT_EXPLORER}/tx/{tx_hash}"


def address_url(address: str) -> str:
    return f"{_explorer or DEFAULT_EXPLORER}/address/{address}"


def get_anchor(sha256_hex: str) -> dict:
    """Read the anchor for a digest. Never raises for "not anchored".

    Safe to call against an unconfigured chain: reports ``available: False`` so
    a read-only view can degrade instead of erroring.
    """
    try:
        contract = _connect()
        timestamp, submitted_by = contract.functions.verify(to_digest(sha256_hex)).call()
        return {
            "available": True,
            "anchored": timestamp > 0,
            "timestamp": timestamp or None,
            "submitted_by": submitted_by,
            "explorer": address_url(contract.address),
        }
    except Exception as exc:
        logger.warning("Anchor lookup failed: %s", exc)
        return {"available": False, "anchored": False, "reason": str(exc)}


def anchor(sha256_hex: str) -> dict:
    """Submit an anchor and return immediately with a pending receipt.

    Does not wait for confirmation: a Sepolia block takes ~12s and no HTTP
    request should block on that. Use `confirm(tx_hash)` later.
    """
    try:
        contract = _connect()
    except NotConfigured as exc:
        return {"status": "skipped", "reason": str(exc)}

    if not _private_key:
        return {"status": "skipped", "reason": "ANCHOR_PRIVATE_KEY is not set."}

    try:
        digest = to_digest(sha256_hex)
        account = _web3.eth.account.from_key(_private_key)

        # Idempotent at the contract level too, but checking first avoids paying
        # for a transaction that is certain to revert.
        if contract.functions.isAnchored(digest).call():
            timestamp, submitted_by = contract.functions.verify(digest).call()
            return {
                "status": "confirmed",
                "already_anchored": True,
                "timestamp": timestamp,
                "submitted_by": submitted_by,
            }

        tx = contract.functions.anchor(digest).build_transaction(
            {
                "from": account.address,
                "nonce": _web3.eth.get_transaction_count(account.address, "pending"),
                "chainId": _chain_id,
                "gas": GAS_LIMIT,
                "maxFeePerGas": _web3.eth.gas_price * 2,
                "maxPriorityFeePerGas": _web3.to_wei(1, "gwei"),
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = _web3.eth.send_raw_transaction(signed.raw_transaction).hex()
        if not tx_hash.startswith("0x"):
            tx_hash = "0x" + tx_hash

        return {
            "status": "pending",
            "tx_hash": tx_hash,
            "explorer": explorer_url(tx_hash),
        }
    except Exception as exc:
        logger.warning("Anchoring failed for %s: %s", sha256_hex[:12], exc)
        return {"status": "failed", "reason": str(exc)}


def confirm(tx_hash: str) -> dict:
    """Check whether a previously submitted anchor has been mined."""
    try:
        _connect()
        receipt = _web3.eth.get_transaction_receipt(tx_hash)
    except Exception as exc:
        # Not mined yet is the common case, not an error.
        return {"status": "pending", "reason": str(exc)}

    if receipt is None:
        return {"status": "pending"}
    if receipt.status == 1:
        return {
            "status": "confirmed",
            "block_number": receipt.blockNumber,
            "explorer": explorer_url(tx_hash),
        }
    return {"status": "failed", "reason": f"transaction reverted in block {receipt.blockNumber}"}
