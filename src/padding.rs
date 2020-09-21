use ordered_float::OrderedFloat;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;
use std::time::SystemTime;

const INFINITY_BIN: f32 = -1f32;

#[derive(Debug, PartialEq, Eq)]
enum State {
    Idle,
    Burst,
    Gap,
}

#[derive(Debug)]
struct WTFPAD {
    config: WTFPADConfig,
    state: State,
    histogram_gap: AdaptiveHistogram,
    histogram_burst: AdaptiveHistogram,
    state_timeout: Option<SystemTime>,
}
#[derive(Debug)]
struct WTFPADConfig {
    padded_size: usize,
}

impl Default for WTFPAD {
    fn default() -> Self {
        WTFPAD::new(
            WTFPADConfig { padded_size: 1500 },
            // Note: this values have been experimentally derived from Tor traffic and have not been adapted to QUIC traffic
            AdaptiveHistogramConfig {
                mu: 0.001_564_1,
                sigma: 0.052_329_6,
                ..Default::default()
            },
            AdaptiveHistogramConfig {
                mu: 0.001_564_1,
                sigma: 0.052_329_6,
                ..Default::default()
            },
        )
    }
}

impl WTFPAD {
    pub fn new(
        config: WTFPADConfig, config_gap: AdaptiveHistogramConfig,
        config_burst: AdaptiveHistogramConfig,
    ) -> WTFPAD {
        WTFPAD {
            config,
            state: State::Idle,
            state_timeout: None,
            histogram_gap: AdaptiveHistogram::new(config_gap),
            histogram_burst: AdaptiveHistogram::new(config_burst),
        }
    }

    /// Simply returns the configured fixed size. The implementation using WTFPAD is responsible for producing a packet of this length.
    fn pad_individual(&self, _size: usize) -> usize {
        self.config.padded_size
    }

    /// Update_state should be called: when a packet is sent/received (depending to the flow being protected), dummy or not.
    fn update_state(&mut self, packet_sent: bool, is_dummy: bool) {
        if packet_sent && !is_dummy {
            // new transmission after a period of silence, switch to Burst mode
            self.state = State::Burst;
        // else, if this function was call by a timeout or by a dummy, check if we should "downgrade" the state
        } else if self.state_timeout.is_some()
            && self.state_timeout.unwrap() < SystemTime::now()
        {
            if self.state == State::Burst {
                self.state = State::Gap;
            } else if self.state == State::Gap {
                self.state = State::Idle;
            }
        }
    }

    /// Returns the time at which the next dummy packet should be sent. Alters self.state_timeout with the returned timeout, if any
    fn next_dummy(&mut self) -> Option<(SystemTime, usize)> {
        match self.state {
            State::Burst => {
                let timeout =
                    timeout_to_future_unixtime(self.histogram_burst.sample());
                self.state_timeout = Some(timeout);
                Some((timeout, self.config.padded_size))
            }
            State::Gap => {
                let timeout =
                    timeout_to_future_unixtime(self.histogram_gap.sample());
                self.state_timeout = Some(timeout);
                Some((timeout, self.config.padded_size))
            }
            State::Idle => None,
        }
    }
}

impl fmt::Display for WTFPAD {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "wtfpad: {:?}", self.state)
    }
}

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
    infinity_bin_config: InfinityBinConfig,
}

impl Default for AdaptiveHistogramConfig {
    fn default() -> Self {
        AdaptiveHistogramConfig {
            n_samples: 100,
            bin_upperbound: 10.0,
            n_bins: 10,
            mu: 1.0,
            sigma: 1.0,
            infinity_bin_config: Default::default(),
        }
    }
}
#[derive(Debug)]
struct InfinityBinConfig {
    burst_length: u32,
    probability_of_fake_burst: f32,
    distribution_type: State,
}

impl Default for InfinityBinConfig {
    fn default() -> Self {
        InfinityBinConfig {
            burst_length: 10,
            probability_of_fake_burst: 0.1,
            distribution_type: State::Idle,
        }
    }
}

/// A histogram that represents an adaptive probability distribution: one can sample/draw a value from this distribution, removing a token from a bin and thus altering the distribution.
/// Once no token is left, the initial distribution is re-computed.
#[derive(Default, Debug)]
struct AdaptiveHistogram {
    config: AdaptiveHistogramConfig,
    histogram: BTreeMap<OrderedFloat<f32>, u32>,
    tokens_left: u32,
}

/// converts a timeout: f32 into UnixTime corresponding to SystemTime::now() + timeout
fn timeout_to_future_unixtime(timeout: f32) -> SystemTime {
    let timeout_ms = (timeout * 1000000.0).ceil() as u64;
    SystemTime::now() + Duration::from_micros(timeout_ms)
}

impl AdaptiveHistogram {
    pub fn new(config: AdaptiveHistogramConfig) -> AdaptiveHistogram {
        let mut ah = AdaptiveHistogram {
            config,
            histogram: BTreeMap::new(),
            tokens_left: 0,
        };
        ah.rebuild();
        ah
    }

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
        selected_bin
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

        // create n_bins+1 exponentially-increasing bins between 0 and bin_upperbound
        let n_bins = self.config.n_bins as usize;
        let mut bins = vec![0f32; n_bins + 1];
        let mut current_bin = self.config.bin_upperbound;
        for i in (1..=n_bins).rev() {
            bins[i] = current_bin;
            current_bin /= 2f32;
        }
        bins[0] = 0.0; // smallest bin is always 0

        self.histogram = BTreeMap::new();
        let mut current_bin_id = 1; // forbid 0 as inter-arrival time
        let mut i = 0;

        // iterate simultaneously on `values` and `bins`, counting the values the values s.t. `bins[i] <= v < bins[i+1]`
        while i < values.len() && current_bin_id < n_bins {
            let mut j = i;
            //consume: `v` while `v < bins[i+1]`, or any `v` if we reached the last bin
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

        // add empty values for unused bins
        while current_bin_id < n_bins {
            self.histogram.insert(OrderedFloat(bins[current_bin_id]), 0);
            current_bin_id += 1;
        }

        // add an "infinity bin" at the end. It probability distribution depends on the state this histogram is modelling
        let infinity_bin_count: u32 = match self
            .config
            .infinity_bin_config
            .distribution_type
        {
            State::Gap => {
                let n = self.config.n_samples as f32;
                let b = self.config.infinity_bin_config.burst_length as f32;
                // The expectation of the geometric distribution of consecutive samples from the histogram to be the average number of packets in a burst `burst_length`.
                // -> the probability of falling into the infinity bin should be p = 1/b
                // Since p = #tokens in infinity bin / #tokens
                // -> #tokens/b = #tokens in infinity bin
                // -> #tokens in infinity bins = #tokens in other bins / (b-1)
                (n / (b - 1.0)).ceil() as u32
            }
            State::Burst => {
                let n = self.config.n_samples as f32;
                //TODO: LB: check, this seems odd
                (n / self.config.infinity_bin_config.probability_of_fake_burst)
                    .ceil() as u32
            }
            State::Idle => 0, // Idle is not modeled by a histogram
        };

        self.histogram
            .insert(OrderedFloat(INFINITY_BIN), infinity_bin_count);

        self.tokens_left = self.config.n_samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_wtpad_states() {
        let mut w: WTFPAD = Default::default();
        assert!(w.state == State::Idle);

        // sending a packet in Idle changes to Burst
        w.state = State::Idle;
        w.update_state(true, false);
        assert!(w.state == State::Burst);

        let some_time_ago: SystemTime =
            SystemTime::now() - Duration::from_millis(100);
        w.state_timeout = Some(some_time_ago);

        // timeout in Burst changes to Gap
        w.state = State::Burst;
        w.update_state(false, false);
        assert!(w.state == State::Gap);

        // timeout in Gap changes to Idle
        w.state = State::Gap;
        w.update_state(false, false);
        assert!(w.state == State::Idle);

        // TODO: this is what the python implementation does, but it seems odd.
        // real packet send in Gap changes to Idle
        w.state = State::Gap;
        w.update_state(false, true);
        assert!(w.state == State::Idle);
    }

    #[test]
    fn test_wtpad_individual_padding() {
        let w: WTFPAD = Default::default();
        assert!(w.pad_individual(1000) == w.config.padded_size);
    }

    #[test]
    fn test_wtpad_drawing_samples() {
        let mut w: WTFPAD = Default::default();
        w.state = State::Burst;
        let dummy = w.next_dummy();
        assert!(dummy.is_some());
    }

    #[test]
    fn test_timeout_to_systemtime() {
        let mut ah: AdaptiveHistogram = Default::default();
        ah.rebuild();

        let timeout_s = ah.sample();
        let timeout_unixtime = timeout_to_future_unixtime(timeout_s);
        let diff = timeout_unixtime.duration_since(SystemTime::now()).unwrap();

        let small_tolerance = 0.001f32; // 1ms
        assert!((timeout_s - diff.as_secs_f32()).abs() < small_tolerance);
    }

    #[test]
    fn test_wtpad() {
        let mut w: WTFPAD = Default::default();
        w.state = State::Burst;
        let dummy = w.next_dummy();
        assert!(dummy.is_some());

        match w.next_dummy() {
            Some(i) => println!("{:?}, {}", i.0, i.1),
            None => println!("No dummy drawn"),
        }
    }
}
