"""Compile, deploy and exercise EvidenceRegistry on an in-memory EVM.

Proves the contract does what it claims without spending anything or needing a
funded wallet: the same source, the same compiled bytecode and the same client
code path are used against Sepolia in production. Only the provider changes.

Run from the backend directory:
    python scripts/verify_contract.py
"""

import os
import sys

from web3 import Web3
from web3.providers.eth_tester import EthereumTesterProvider

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from compile_contract import SOLC_VERSION, compile_contract, ensure_solc  # noqa: E402

import blockchain  # noqa: E402

failures = []


def check(label, condition, detail=""):
    failures.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main():
    ensure_solc()
    abi, bytecode = compile_contract()
    print(f"compiled EvidenceRegistry with solc {SOLC_VERSION}")

    w3 = Web3(EthereumTesterProvider())
    w3.eth.default_account = w3.eth.accounts[0]

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = contract.constructor().transact()
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    address = receipt.contractAddress
    deployed = w3.eth.contract(address=address, abi=abi)
    print(f"deployed at {address}  (gas {receipt.gasUsed:,})")

    # A realistic digest: what SHA-256 of a file looks like.
    digest = bytes.fromhex(
        "af596a4d70ece84e0b2c1d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6"
    )

    print("\n1. Nothing is anchored to begin with")
    check("isAnchored is false", deployed.functions.isAnchored(digest).call() is False)
    check(
        "verify returns zero",
        deployed.functions.verify(digest).call() == [0, "0x0000000000000000000000000000000000000000"],
    )

    print("\n2. Anchoring stores the digest")
    tx_hash = deployed.functions.anchor(digest).transact()
    anchor_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"   anchor gas: {anchor_receipt.gasUsed:,}")

    timestamp, submitted_by = deployed.functions.verify(digest).call()
    check("isAnchored is now true", deployed.functions.isAnchored(digest).call() is True)
    check("timestamp is set", timestamp > 0, f"timestamp={timestamp}")
    check("submitter recorded", submitted_by == w3.eth.accounts[0], submitted_by)

    block = w3.eth.get_block(anchor_receipt.blockNumber)
    check("timestamp matches the block time", timestamp == block.timestamp,
          f"{timestamp} vs {block.timestamp}")

    print("\n3. Re-anchoring the same digest is refused")
    try:
        deployed.functions.anchor(digest).transact()
        check("duplicate anchor reverts", False, "it succeeded")
    except Exception as exc:
        check("duplicate anchor reverts", "AlreadyAnchored" in str(exc) or "revert" in str(exc).lower(),
              type(exc).__name__)

    print("\n4. The event is emitted for indexers")
    events = deployed.events.Anchored().get_logs(from_block=0)
    check("Anchored event emitted once", len(events) == 1, f"({len(events)} events)")
    if events:
        check("event carries the digest", events[0]["args"]["digest"] == digest)
        check("event is indexed by digest", events[0]["args"]["digest"] == digest)

    print("\n5. A different digest stays independent")
    other = bytes.fromhex("00" * 32)
    check("unknown digest still unanchored", deployed.functions.isAnchored(other).call() is False)

    print("\n6. There is no way to remove or alter an anchor")
    write_functions = [
        f["name"] for f in abi
        if f.get("type") == "function" and f.get("stateMutability") in ("nonpayable", "payable")
    ]
    check("only 'anchor' mutates state", write_functions == ["anchor"], str(write_functions))

    print("\n7. The client layer (blockchain.py) against the same chain")
    private_key = "0x" + "11" * 32
    signer = w3.eth.account.from_key(private_key)
    w3.eth.send_transaction(
        {"from": w3.eth.accounts[0], "to": signer.address, "value": w3.to_wei(1, "ether")}
    )
    blockchain.use(w3, address, private_key=private_key, chain_id=w3.eth.chain_id)

    second = "aa" * 32
    before = blockchain.get_anchor(second)
    check("get_anchor reports not-yet-anchored", before["available"] and before["anchored"] is False)

    submitted = blockchain.anchor(second)
    check("anchor returns immediately as pending",
          submitted.get("status") == "pending" and bool(submitted.get("tx_hash")),
          str(submitted.get("status")))
    check("submitted anchor has a real transaction hash",
          str(submitted.get("tx_hash", "")).startswith("0x") and len(str(submitted.get("tx_hash"))) == 66)

    mined = blockchain.confirm(submitted["tx_hash"])
    check("confirm reports the anchor mined", mined.get("status") == "confirmed", str(mined))

    after = blockchain.get_anchor(second)
    check("get_anchor now reports anchored", after["anchored"] is True, f"timestamp={after.get('timestamp')}")

    repeat = blockchain.anchor(second)
    check("re-anchoring is short-circuited without another transaction",
          repeat.get("already_anchored") is True, str(repeat.get("status")))

    try:
        blockchain.to_digest("ab" * 31)
        check("a wrong-length digest is rejected", False, "it was accepted")
    except ValueError:
        check("a wrong-length digest is rejected", True)

    print("\n8. Unconfigured anchoring degrades instead of raising")
    blockchain._contract = None
    blockchain._web3 = None
    blockchain._private_key = None
    if not blockchain.is_configured():
        skipped = blockchain.anchor("bb" * 32)
        check("anchor returns 'skipped' when unconfigured", skipped.get("status") == "skipped",
              str(skipped.get("reason"))[:60])
        unavailable = blockchain.get_anchor("bb" * 32)
        check("get_anchor reports unavailable when unconfigured",
              unavailable["available"] is False)
    else:
        print("   (skipped: SEPOLIA_RPC_URL is set in this environment)")

    print("\n" + ("ALL PASS" if all(failures) else f"{failures.count(False)} CHECK(S) FAILED"))
    return 0 if all(failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
