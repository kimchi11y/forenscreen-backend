"""Compile contracts/EvidenceRegistry.sol and write the artifact.

Writes contracts/EvidenceRegistry.json containing the ABI and creation bytecode.
blockchain.py loads the ABI at runtime, and scripts/deploy_contract.py needs the
bytecode, so run this once after changing the contract:

    python scripts/compile_contract.py
"""

import json
import os
import sys

import solcx

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT_PATH = os.path.join(BACKEND_DIR, "contracts", "EvidenceRegistry.sol")
ARTIFACT_PATH = os.path.join(BACKEND_DIR, "contracts", "EvidenceRegistry.json")
SOLC_VERSION = "0.8.24"


def ensure_solc():
    """Install the pinned compiler if it is not already present."""
    try:
        installed = [str(v) for v in solcx.get_installed_solc_versions()]
    except Exception:
        installed = []
    if SOLC_VERSION not in installed:
        print(f"installing solc {SOLC_VERSION} (one time)...")
        solcx.install_solc(SOLC_VERSION)
    solcx.set_solc_version(SOLC_VERSION)


def compile_contract():
    compiled = solcx.compile_files(
        [CONTRACT_PATH],
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
        optimize=True,
    )
    key = next(k for k in compiled if k.endswith(":EvidenceRegistry"))
    return compiled[key]["abi"], compiled[key]["bin"]


def write_artifact(abi, bytecode):
    with open(ARTIFACT_PATH, "w") as handle:
        json.dump({"abi": abi, "bytecode": bytecode, "solc": SOLC_VERSION}, handle, indent=2)
    return ARTIFACT_PATH


def main():
    ensure_solc()
    abi, bytecode = compile_contract()
    path = write_artifact(abi, bytecode)
    functions = [f["name"] for f in abi if f.get("type") == "function"]
    print(f"compiled with solc {SOLC_VERSION}")
    print(f"functions: {functions}")
    print(f"wrote {os.path.relpath(path, BACKEND_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
