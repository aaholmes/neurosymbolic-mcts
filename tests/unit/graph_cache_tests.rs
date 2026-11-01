//! CUDA graph replay of the half-precision forward pass
//! (`kingfisher::nn::graph_cache`): batches are padded to a few fixed sizes, and
//! replaying a recorded graph must give what the network computes directly.

#![cfg(feature = "neural")]

use kingfisher::nn::graph_cache::bucket;
use std::collections::BTreeSet;

#[test]
fn batches_are_padded_to_eight_then_to_multiples_of_sixteen() {
    let cap = 128;
    let sizes: BTreeSet<usize> = (1..=cap).map(|b| bucket(b, cap)).collect();
    assert_eq!(
        sizes.into_iter().collect::<Vec<_>>(),
        [8, 16, 32, 48, 64, 80, 96, 112, 128]
    );
    for b in 1..=cap {
        let k = bucket(b, cap);
        assert!(b <= k && k <= cap, "batch {b} -> {k}");
    }
    // A cap that is not a multiple of 16 is its own largest size.
    assert_eq!(bucket(99, 100), 100);
    assert_eq!(bucket(100, 100), 100);
    assert_eq!(bucket(3, 4), 4);
}

// Replay on the GPU is tested in `tests/cuda_graph_tests.rs`, in a process of
// its own: other tests' CUDA calls (creating and dropping devices, streams and
// cuDNN handles) can make a recording fail.
