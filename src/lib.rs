use supervised_learning::Classifier;
use hash_histogram::{KeyType, mode_values};
use std::cmp::Ordering;
use std::sync::Arc;
use trait_set::trait_set;
use kmeans::Kmeans;

trait_set! {
    pub trait LabelType = KeyType + Ord;
}

pub struct Knn<L: LabelType, T, V, D: Fn(&T,&T) -> V> {
    k: usize,
    examples: Vec<(L,T)>,
    distance: Arc<D>,
}

impl<L: LabelType, T, V, D: Fn(&T,&T) -> V> Knn<L, T, V, D> {
    pub fn new(k: usize, distance: Arc<D>) -> Self {
        Knn {k, examples: Vec::new(), distance}
    }

    pub fn add_example(&mut self, example: (L, T)) {
        self.examples.push(example);
    }
}

impl<L: LabelType, T: Clone, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V> Classifier<T,L> for Knn<L, T, V, D> {
    fn train(&mut self, training_images: &Vec<(L,T)>) {
        for img in training_images {
            // TODO: Bug report: self.add_example(img.clone()); // Flagged as type error by IDE, but compiles fine.
            self.add_example((img.0.clone(), img.1.clone()));
        }
    }

    fn classify(&self, example: &T) -> L {
        let mut distances: Vec<(V, L)> = self.examples.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0.clone()))
            .collect();
        distances.sort_by(cmp_w_label);
        let iter = distances.iter().take(self.k).map(|(_, label)| label.clone());
        mode_values(iter).unwrap()
    }
}

fn cmp_w_label<L: LabelType, V: Copy + PartialEq + PartialOrd>(a: &(V, L), b: &(V, L)) -> Ordering {
    cmp_f64(&a.0, &b.0)
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

pub struct ClusteredKnn<L: LabelType, T, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V, M: Fn(&Vec<&T>) -> T> {
    knn: Knn<L, T, V, D>,
    clusters: Option<Kmeans<T, V, D>>,
    num_clusters: usize,
    mean: Arc<M>
}

impl <L: LabelType, T, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V, M: Fn(&Vec<&T>) -> T> ClusteredKnn<L, T, V, D, M> {
    pub fn new(k: usize, num_clusters: usize, distance: Arc<D>, mean: Arc<M>) -> Self {
        Self {knn: Knn::new(k, distance), clusters: None, num_clusters, mean}
    }
}

impl<L: LabelType, T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V, M: Fn(&Vec<&T>) -> T> Classifier<T,L> for ClusteredKnn<L, T, V, D, M> {
    fn train(&mut self, training_images: &Vec<(L,T)>) {
        let data = training_images.iter().map(|(_,t)| t.clone()).collect::<Vec<_>>();
        let clusters = Kmeans::new(self.num_clusters, &data, self.knn.distance.clone(), self.mean.clone());
        let mut labeler = Knn::new(self.knn.k, self.knn.distance.clone());
        labeler.train(training_images);
        for example in clusters.copy_means() {
            let label = labeler.classify(&example);
            self.knn.add_example((label, example));
        }
        self.clusters = Some(clusters);
    }

    fn classify(&self, example: &T) -> L {
        let best = self.clusters.as_ref().unwrap().best_matching_mean(example);
        self.knn.classify(&best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn() {
        let labeled_data = vec![(0, 5.0), (0, 2.0), (1, 1.0), (1, 3.0)];
        let mut classifier = Knn::new(3, Arc::new(|f1: &f64, f2: &f64| (f1 - f2).abs()));
        classifier.train(&labeled_data);
        for (label, value) in [(0, 10.0), (1, 0.0), (0, 3.3), (1, 2.9)].iter() {
            let classification = classifier.classify(value);
            println!("{} ({}): classified as {}", value, label, classification);
            assert_eq!(*label, classification);
        }
    }

    fn manhattan(n1: &i32, n2: &i32) -> i32 {
        (n1 - n2).abs()
    }

    fn mean(nums: &Vec<&i32>) -> i32 {
        let total: i32 = nums.iter().map(|i| *i).sum();
        total / (nums.len() as i32)
    }

    #[test]
    fn test_clustered() {
        let mut examples = vec![];
        for i in 0..100 {
            examples.push((if i < 50 {0} else {1}, i));
        }
        for _ in 0..20 {
            let mut classifier = ClusteredKnn::new(3, 4, Arc::new(manhattan), Arc::new(mean));
            classifier.train(&examples);
            for i in 0..100 {
                if i < 25 || i > 75 {
                    assert_eq!(if i < 50 {0} else {1}, classifier.classify(&i));
                }
            }
        }
    }
}
