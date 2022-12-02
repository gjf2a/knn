use supervised_learning::Classifier;
use hash_histogram::HashHistogram;
use std::cmp::Ordering;
use std::sync::Arc;

#[derive(Clone)]
pub struct Knn<T, V, D: Fn(&T,&T) -> V> {
    k: usize,
    examples: Vec<(u8,T)>,
    distance: Arc<D>,
}

impl<T, V, D: Fn(&T,&T) -> V> Knn<T, V, D> {
    pub fn new(k: usize, distance: Arc<D>) -> Knn<T, V, D> {
        Knn {k, examples: Vec::new(), distance}
    }

    pub fn add_example(&mut self, img: (u8, T)) {
        self.examples.push(img);
    }
}

impl<T: Clone, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V> Classifier<T> for Knn<T, V, D> {
    fn train(&mut self, training_images: &Vec<(u8,T)>) {
        for img in training_images {
            self.add_example((img.0, img.1.clone()));
        }
    }

    fn classify(&self, example: &T) -> u8 {
        let mut distances: Vec<(V, u8)> = self.examples.iter()
            .map(|img| ((self.distance)(example, &img.1), img.0))
            .collect();
        distances.sort_by(cmp_f64);

        let mut labels = HashHistogram::new();
        for item in distances.iter().take(self.k) {
            labels.bump(item.1);
        }
        labels.mode()
    }
}

// Borrowed from: https://users.rust-lang.org/t/sorting-vector-of-vectors-of-f64/16264
fn cmp_f64<V: Copy + PartialEq + PartialOrd>(a: &V, b: &V) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    return Ordering::Equal;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        // Basically, "assert false"
        assert_eq!(2 + 2, 5);
    }
}
