use ordered_float::OrderedFloat;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::BTreeMap;
use std::fmt;

/// AdaptiveHistogramConfig characterizes the initial state of an AdaptiveHistogram.
/// n_samples, bin_upperbound and n_bins relate to the precision used when building the histogram.
///  - n_samples dictates how many samples we take to rebuild the distribution;
///  - bin_upperbound is the "cutoff" after which the tail of the distribution is represented with a single bucket;
///  - n_bins is the granuarity of the distribution;
/// Higher values increase computation time and precision.
/// mu and sigma represent the mean and standard deviation of the target Normal distribution.
/// These two values are of paramount importance for generating dummies at the right time and should be carefully tuned to match the target traffic.
#[derive(Debug)]
struct AdaptiveHistogramConfig {
    n_samples: u32,
    bin_upperbound: f32,
    n_bins: u32,
    mu: f32,
    sigma: f32,
}

impl Default for AdaptiveHistogramConfig {
    fn default() -> Self {
        AdaptiveHistogramConfig {
            n_samples: 100,
            bin_upperbound: 10f32,
            n_bins: 10,
            // Note: this values have been experimentally derived from Tor traffic and have not been adapted to QUIC traffic
            mu: 0.001564159,
            sigma: 0.052329599,
        }
    }
}
/// A histogram that represents an adaptive probability distribution: one can sample/draw a value from this distribution, removing a token from a bin and thus altering the distribution.
/// Once no token is left, the initial distribution is re-computed.
/// ```
/// let mut ah: AdaptiveHistogram = Default::default();
/// ah.rebuild();
/// println!("{}", ah);
/// println!("Value sampled: {}", ah.sample());
/// ```
#[derive(Default, Debug)]
struct AdaptiveHistogram {
    config: AdaptiveHistogramConfig,
    histogram: BTreeMap<OrderedFloat<f32>, u32>,
    tokens_left: u32,
}

impl AdaptiveHistogram {
    /// Samples a value from the probability distribution, altering it. If the "distribution is depleted" (all tokens have been drawn), this recomputes a fresh distribution from the configuration.
    fn sample(&mut self) -> f32 {
        if self.tokens_left < 1 {
            self.rebuild()
        }

        let mut kv = (OrderedFloat(0f32), 1);
        let mut target_token = Uniform::from(1..self.tokens_left + 1)
            .sample(&mut rand::thread_rng())
            as i32;
        for (&bin, &count) in &self.histogram {
            target_token -= count as i32;
            if target_token <= 0 {
                kv = (bin, count);
                break;
            }
        }
        self.histogram.insert(kv.0, kv.1 - 1);
        let selected_bin = kv.0.into();
        self.tokens_left -= 1;
        return selected_bin;
    }

    /// Build a histogram by drawing values from the normal distribution and placing them in buckets. O(N) space, O(N log(N)) time where N is the number of samples
    fn rebuild(&mut self) {
        // draw n_samples values from Norm(mu, sigma)
        let normal = Normal::new(self.config.mu, self.config.sigma).unwrap();
        let mut values: Vec<f32> = normal
            .sample_iter(&mut rand::thread_rng())
            .filter(|&v| v > 0f32)
            .take(self.config.n_samples as usize)
            .collect();
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        // create n_bins exponentially-increasing bins between 0 and bin_upperbound
        let n_bins = self.config.n_bins as usize;
        let mut bins = vec![0f32; n_bins];
        let mut current_bin = self.config.bin_upperbound;
        for i in (0..n_bins).rev() {
            bins[i] = current_bin;
            current_bin /= 2f32;
        }
        bins[0] = 0f32; // smallest bin is always 0

        self.histogram = BTreeMap::new();
        let mut current_bin_id = 0;
        let mut i = 0;

        // iterate simultaneously on `values` and `bins`, counting the values the values s.t. bins[i] <= v < bins[i+1]
        while i < values.len() && current_bin_id < n_bins {
            let mut j = i;
            //consume: v while v < bins[i+1], or any v if we reached the last bin
            while j < values.len()
                && (current_bin_id == n_bins - 1
                    || values[j] < bins[current_bin_id + 1])
            {
                j += 1;
            }
            let count = (j - i) as u32;
            self.histogram
                .insert(OrderedFloat(bins[current_bin_id]), count);

            current_bin_id += 1;
            i = j;
        }
        // add empty values for unused bins$
        while current_bin_id < n_bins {
            self.histogram.insert(OrderedFloat(bins[current_bin_id]), 0);
            current_bin_id += 1;
        }
        self.tokens_left = self.config.n_samples;
    }
}

impl fmt::Display for AdaptiveHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok({
            for (bin, count) in &self.histogram {
                writeln!(f, "{}: {}", bin, count).unwrap();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_adaptive_histogram() {
        let mut ah: AdaptiveHistogram = Default::default();
        assert!(ah.tokens_left == 0);

        ah.config.n_samples = 100;
        ah.rebuild();

        // sample n token
        for i in 0..ah.config.n_samples {
            // check that token_left is updated accordingly
            assert!(ah.config.n_samples - i == ah.tokens_left);

            // check that the token is valid
            let token = ah.sample();
            assert!(0f32 <= token && token <= ah.config.bin_upperbound);
        }

        // next sample() will trigger rebuild() internally
        let token = ah.sample();
        assert!(0f32 <= token && token <= ah.config.bin_upperbound);
        assert!(ah.tokens_left == ah.config.n_samples - 1);
    }
}
