use std::time::Duration;

pub(super) const MAX_BATCH_SAMPLES: usize = 64;

/// Bound queued GPU work so cancellation stays responsive. GPU timings exclude
/// readback and display; a slow Blender callback must not shrink the batch.
pub(super) struct SampleBatches {
    samples: u32,
    limit: u32,
    target_ms: f32,
}

impl SampleBatches {
    pub(super) fn new(preview_interval: Option<Duration>) -> Self {
        let interval = preview_interval.unwrap_or(Duration::from_millis(50));
        Self {
            samples: 1, // Show the first sample without waiting for a batch.
            limit: if interval.is_zero() {
                1 // Explicit every-sample previews are useful for diagnostics.
            } else {
                MAX_BATCH_SAMPLES as u32
            },
            target_ms: interval.as_secs_f32().min(0.05) * 1000.0,
        }
    }

    pub(super) fn next_count(&self, remaining: u32) -> u32 {
        self.samples.min(remaining)
    }

    pub(super) fn record_gpu_time(&mut self, samples: u32, elapsed_ms: f32) {
        let desired = (self.target_ms * samples as f32 / elapsed_ms.max(0.001)) as u32;
        // Grow gradually, but shrink immediately if sample cost increases.
        self.samples = desired.clamp(1, self.limit).min(samples.saturating_mul(2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_samples_batch_without_skipping_the_first_or_last() {
        let mut batches = SampleBatches::new(Some(Duration::from_millis(100)));
        let mut completed = 0;
        let mut counts = Vec::new();
        while completed < 250 {
            let count = batches.next_count(250 - completed);
            assert!((1..=64).contains(&count));
            counts.push(count);
            completed += count;
            batches.record_gpu_time(count, count as f32 * 0.1);
        }
        assert_eq!(counts[0], 1);
        assert!(counts.contains(&64));
        assert_eq!(completed, 250);
        assert_eq!(batches.next_count(0), 0);
    }

    #[test]
    fn slow_samples_shrink_to_one_and_growth_is_bounded() {
        let mut batches = SampleBatches::new(None);
        batches.record_gpu_time(1, 0.0);
        assert_eq!(batches.next_count(100), 2);
        batches.record_gpu_time(2, 160.0);
        assert_eq!(batches.next_count(100), 1);
    }

    #[test]
    fn short_preview_interval_limits_queued_work() {
        let mut batches = SampleBatches::new(Some(Duration::from_millis(10)));
        for _ in 0..8 {
            let count = batches.next_count(100);
            batches.record_gpu_time(count, count as f32 * 2.0);
        }
        assert_eq!(batches.next_count(100), 5);
    }

    #[test]
    fn zero_interval_preserves_every_sample_previews() {
        let mut batches = SampleBatches::new(Some(Duration::ZERO));
        for _ in 0..8 {
            batches.record_gpu_time(1, 0.01);
            assert_eq!(batches.next_count(100), 1);
        }
    }
}
