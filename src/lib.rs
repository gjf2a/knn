use supervised_learning::Classifier;
use hash_histogram::{KeyType, mode_values};
use std::cmp::Ordering;
use std::sync::Arc;

pub struct Knn<L: KeyType, I, M, D: Fn(&I,&I) -> M> {
    k: usize,
    images: Vec<(u8,I)>,
    distance: D,
}

impl<I, M, D: Fn(&I,&I) -> M> Knn<I, M, D> {
    pub fn new(k: usize, distance: D) -> Knn<I, M, D> {
        Knn {k, images: Vec::new(), distance}
    }

    pub fn add_example(&mut self, img: (u8, I)) {
        self.images.push(img);
    }
}

impl<I: Clone, M: Copy + PartialEq + PartialOrd, D: Fn(&I,&I) -> M> Classifier<I> for Knn<I, M, D> {
    fn train(&mut self, training_images: &Vec<(u8,I)>) {
        for img in training_images {
            // TODO: Bug report: self.add_example(img.clone()); // Flagged as type error by IDE, but compiles fine.
            self.add_example((img.0, img.1.clone()));
        }
    }

    fn classify(&self, example: &I) -> u8 {
        let mut distances: Vec<(M, u8)> = self.images.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0))
            .collect();
        distances.sort_by(cmp_f64);
        let iter = distances.iter().take(self.k).map(|(_, label)| label.clone());
        mode_values(iter).unwrap()
    }
}

// Borrowed from: https://users.rust-lang.org/t/sorting-vector-of-vectors-of-f64/16264
fn cmp_f64<M: Copy + PartialEq + PartialOrd>(a: &M, b: &M) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let labeled_data = vec![(0, 5.0), (0, 2.0), (1, 1.0), (1, 3.0)];
        let mut classifier = Knn::new(3, |f1: &f64, f2: &f64| (f1 - f2).abs());
        classifier.train(&labeled_data);
        for (label, value) in [(0, 10.0), (1, 0.0), (0, 3.3), (1, 2.9)].iter() {
            let classification = classifier.classify(value);
            println!("{} ({}): classified as {}", value, label, classification);
            assert_eq!(*label, classification);
        }
    }
}
