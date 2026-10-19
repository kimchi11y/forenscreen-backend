"""Deploy EvidenceRegistry to a chain and print the address to configure.

Reads from the environment:

    SEPOLIA_RPC_URL        an RPC endpoint (Alchemy, Infura, a public node, ...)
    ANCHOR_PRIVATE_KEY     a THROWAWAY key holding a little test ETH

Nothing personal should ever be used here: this key signs every anchor, and a
testnet key in a .env file is a testnet key in a .env file.

    python scripts/deploy_contract.py
"""

import json
import os
import sys

from web3 import Web3

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_PATH = os.path.join(BACKEND_DIR, "contracts", "EvidenceRegistry.json")
GAS_LIMIT = 1_500_000


def main():
    rpc_url = os.getenv("SEPOLIA_RPC_URL")
    private_key = os.getenv("ANCHOR_PRIVATE_KEY")

    if not rpc_url:
        sys.exit("SEPOLIA_RPC_URL is not set. See .env.example.")
    if not private_key:
        sys.exit("ANCHOR_PRIVATE_KEY is not set. See .env.example.")

    try:
        with open(ARTIFACT_PATH) as handle:
            artifact = json.load(handle)
    except FileNotFoundError:
        sys.exit("Contract artifact missing. Run scripts/compile_contract.py first.")

    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        sys.exit(f"Could not reach {rpc_url}.")

    account = w3.eth.account.from_key(private_key)
    balance = w3.eth.get_balance(account.address)
    print(f"chain id : {w3.eth.chain_id}")
    print(f"deployer : {account.address}")
    print(f"balance  : {w3.from_wei(balance, 'ether')} ETH")

    if balance == 0:
        sys.exit(
            "Deployer has no ETH. Get Sepolia test ETH from a faucet first "
            "(a gated faucet usually needs a small mainnet balance or an "
            "Alchemy/Infura account)."
        )

    factory = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    tx = factory.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address, "pending"),
            "chainId": w3.eth.chain_id,
            "gas": GAS_LIMIT,
            "maxFeePerGas": w3.eth.gas_price * 2,
            "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
        }
    )

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"\nsubmitted {tx_hash.hex()}")
    print("waiting for a receipt (a Sepolia block takes ~12s)...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    if receipt.status != 1:
        sys.exit(f"Deployment reverted in block {receipt.blockNumber}.")

    address = receipt.contractAddress
    print(f"\ndeployed at {address}  (gas {receipt.gasUsed:,})")
    print(f"block {receipt.blockNumber}")
    print("\nAdd this to backend/.env:")
    print(f"REGISTRY_CONTRACT_ADDRESS={address}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
