use supervised_learning::Classifier;
use hash_histogram::{KeyType, mode_values};
use std::cmp::Ordering;
use std::sync::Arc;
use trait_set::trait_set;
use kmeans::Kmeans;

trait_set! {
    pub trait LabelType = KeyType + Ord;
}

pub struct Knn<L: LabelType, T, V> {
    k: usize,
    examples: Vec<(L,T)>,
    distance: Arc<fn(&T,&T) -> V>,
}

impl<L: LabelType, T: Clone, V: Copy + PartialEq + PartialOrd> Clone for  Knn<L, T, V> {
    fn clone(&self) -> Self {
        Self { k: self.k.clone(), examples: self.examples.clone(), distance: self.distance.clone() }
    }
}

impl<L: LabelType, T, V> Knn<L, T, V> {
    pub fn new(k: usize, distance: Arc<fn(&T,&T) -> V>) -> Self {
        Knn {k, examples: Vec::new(), distance}
    }

    pub fn clear_examples(&mut self) {
        self.examples.clear();
    }

    pub fn has_enough_examples(&self) -> bool {
        self.examples.len() >= self.k
    }

    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn add_example(&mut self, example: (L, T)) {
        self.examples.push(example);
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }
}

impl<L: LabelType, T: Clone, V: Copy + PartialEq + PartialOrd> Classifier<T,L> for Knn<L, T, V> {
    fn train(&mut self, training_images: &Vec<(L,T)>) {
        for img in training_images {
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

pub struct ClusteredKnn<L: LabelType, T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd> {
    knn: Knn<L, T, V>,
    clusters: Option<Kmeans<T, V>>,
    num_clusters: usize,
    mean: Arc<fn(&Vec<&T>) -> T>
}

impl<L: LabelType, T: Clone + PartialEq, V: Copy + Clone + PartialEq + PartialOrd + Into<f64>> Clone for ClusteredKnn<L, T, V> {
    fn clone(&self) -> Self {
        Self { knn: self.knn.clone(), clusters: self.clusters.clone(), num_clusters: self.num_clusters.clone(), mean: self.mean.clone() }
    }
}    

impl <L: LabelType, T: Clone + PartialEq, V: Copy + Clone + PartialEq + PartialOrd + Into<f64>> ClusteredKnn<L, T, V> {
    pub fn new(k: usize, num_clusters: usize, distance: Arc<fn(&T,&T) -> V>, mean: Arc<fn(&Vec<&T>) -> T>) -> Self {
        Self {knn: Knn::new(k, distance), clusters: None, num_clusters, mean}
    }

    pub fn train_from_clusters(&mut self, clusters: &Kmeans<T, V>, training_examples: &Vec<(L,T)>) {
        let clusters: Kmeans<T, V> = clusters.clone();
        
        let mut labeler = Knn::new(self.knn.k, self.knn.distance.clone());
        labeler.train(training_examples);
        for example in clusters.copy_means() {
            let label = labeler.classify(&example);
            self.knn.add_example((label, example));
        }
        self.clusters = Some(clusters.clone());
    }
}

impl<L: LabelType, T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>> Classifier<T,L> for ClusteredKnn<L, T, V> {
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
        assert_eq!(classifier.get_k(), 3);
        classifier.train(&labeled_data);
        assert_eq!(classifier.len(), labeled_data.len());
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
