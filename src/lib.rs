use supervised_learning::Classifier;
use hash_histogram::HashHistogram;
use std::cmp::Ordering;
use std::hash::Hash;

pub struct Knn<L: Clone, I, M, D: Fn(&I,&I) -> M> {
    k: usize,
    images: Vec<(L,I)>,
    distance: D,
}

impl<L: Clone, I, M, D: Fn(&I,&I) -> M> Knn<L, I, M, D> {
    pub fn new(k: usize, distance: D) -> Knn<L, I, M, D> {
        Knn {k, images: Vec::new(), distance}
    }

    pub fn add_example(&mut self, img: (L, I)) {
        self.images.push(img);
    }
}

impl<L: Clone+Ord+Hash+Eq, I: Clone, M: Copy + PartialEq + PartialOrd, D: Fn(&I,&I) -> M> Classifier<I,L> for Knn<L, I, M, D> {
    fn train(&mut self, training_images: &Vec<(L,I)>) {
        for img in training_images {
            self.add_example((img.0.clone(), img.1.clone()));
        }
    }

    fn classify(&self, example: &I) -> L {
        let mut distances: Vec<(M, L)> = self.images.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0.clone()))
            .collect();
        distances.sort_by(cmp_f64);

        let mut labels = HashHistogram::new();
        for item in distances.iter().take(self.k) {
            labels.bump(&item.1);
        }
        labels.mode().unwrap().0
    }
}

// Borrowed from: https://users.rust-lang.org/t/sorting-vector-of-vectors-of-f64/16264
fn cmp_f64<M: PartialEq + PartialOrd>(a: &M, b: &M) -> Ordering {
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
            println!("test: {} ({}) classified as {}", value, label, classification);
            assert_eq!(*label, classification);
        }
    }
}
