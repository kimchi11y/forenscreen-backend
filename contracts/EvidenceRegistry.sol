// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ForenScreen EvidenceRegistry
/// @notice Records that a SHA-256 digest existed at or before a block time.
///
/// @dev What this contract is, and is not.
///
/// It is a notary. It proves a digest existed by a given time, and that nobody -
/// including the deployer - has since altered or removed that record. That is
/// the whole point of anchoring: a third party can verify it without trusting
/// ForenScreen's database, its server, or its operators.
///
/// It is not a chain of custody, and it does not make evidence authentic. A hash
/// is content-blind: a doctored image hashes perfectly well. This contract
/// cannot tell whether the file behind a digest is genuine, who the submitter
/// really was (msg.sender is a key, not a person), or whether the same image was
/// registered here first by someone else. Those questions are answered by the
/// off-chain custody log, and the trust model is documented in Chapter 4.
///
/// Deliberately absent: no owner, no admin role, no upgrade path, no deletion.
/// Every one of those would reintroduce exactly the trust assumption the
/// contract exists to remove.
contract EvidenceRegistry {
    struct Anchor {
        // 0 means "never anchored" - block.timestamp is never 0 in practice.
        uint64 timestamp;
        address submittedBy;
    }

    mapping(bytes32 => Anchor) private _anchors;

    /// @param digest     SHA-256 of the file, as 32 raw bytes.
    /// @param submittedBy The account that paid for the anchor.
    /// @param timestamp  Block time the anchor was mined.
    event Anchored(bytes32 indexed digest, address indexed submittedBy, uint64 timestamp);

    error AlreadyAnchored(bytes32 digest);

    /// @notice Anchor a digest. Reverts if it is already anchored.
    /// @dev Idempotence is enforced here rather than by the caller: a second
    ///      anchor of the same digest would add nothing, and reverting makes the
    ///      duplicate attempt explicit instead of silently costly.
    function anchor(bytes32 digest) external {
        if (_anchors[digest].timestamp != 0) {
            revert AlreadyAnchored(digest);
        }
        _anchors[digest] = Anchor({
            timestamp: uint64(block.timestamp),
            submittedBy: msg.sender
        });
        emit Anchored(digest, msg.sender, uint64(block.timestamp));
    }

    /// @notice Look up a digest.
    /// @return timestamp   Block time of the anchor, or 0 if never anchored.
    /// @return submittedBy Account that anchored it, or address(0).
    function verify(bytes32 digest) external view returns (uint64 timestamp, address submittedBy) {
        Anchor memory record = _anchors[digest];
        return (record.timestamp, record.submittedBy);
    }

    /// @notice Whether a digest has been anchored.
    function isAnchored(bytes32 digest) external view returns (bool) {
        return _anchors[digest].timestamp != 0;
    }
}
