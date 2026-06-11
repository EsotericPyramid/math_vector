use std::{
    error::Error,
    fs,
    io,
    hash::{Hash, Hasher, DefaultHasher},
};

use std::path::{Path, PathBuf};


use crate::vector::{
    VectorOps,
    vec_util_traits::Get, 
    vector_initialization::UninitVectorExpr, 
    vector_math::ConcreteVectorExpr
};

pub trait AsText: Sized {
    type Error: Error;

    fn as_text(&self) -> String;
    fn as_text_utf8_bytes(&self) -> Vec<u8> {
        self.as_text().into_bytes()
    }
    fn from_text(text: &str) -> Result<Self, Self::Error>;
}

pub trait AsData<E: Error>: Sized {
    type Error: Error;

    fn as_data(&self) -> Vec<u8>;
    fn from_data(data: &[u8]) -> Result<Self, Self::Error>;
}

#[derive(Debug)]
#[cfg(feature = "csv")]
pub enum CSVError<FieldError: Error> {
    CSVError(csv::Error),
    CellOutOfBounds,
    FieldError(FieldError),
    FileError(io::Error),
}

impl<FieldError: Error> std::fmt::Display for CSVError<FieldError> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CSVError(e) => write!(f, "{}", e),
            Self::CellOutOfBounds => write!(f, "CellOutOfBounds"),
            Self::FieldError(e) => write!(f, "{}", e),
            Self::FileError(e) => write!(f, "{}", e),
        }
    }
}

impl<FieldError: Error> From<csv::Error> for CSVError<FieldError> {
    fn from(value: csv::Error) -> Self {
        Self::CSVError(value)
    }
}

impl<FieldError: Error> From<io::Error> for CSVError<FieldError> {
    fn from(value: io::Error) -> Self {
        Self::FileError(value)
    }
}



impl<FieldError: Error> Error for CSVError<FieldError> {}


fn generate_tmp_file_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let mut temp_file = std::env::temp_dir();
    let mut hasher = DefaultHasher::new();
    path.as_ref().hash(&mut hasher);
    temp_file.push(format!("MathVectorTemp{:016X}", hasher.finish()));
    temp_file
}

struct IterInsert<I: Iterator> where I::Item: Clone {
    idx: usize,
    iter: I,
    insertion_item: Option<I::Item>,
    insertion_idx: usize,
    filler: I::Item,
}

impl<I: Iterator> IterInsert<I> where I::Item: Clone {
    fn new(iter: I, item: I::Item, idx: usize, filler: I::Item) -> Self {
        Self {
            idx: 0,
            iter,
            insertion_item: Some(item),
            insertion_idx: idx,
            filler,
        }
    }
}

impl<I: Iterator> Iterator for IterInsert<I> where I::Item: Clone {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let normal_item = self.iter.next();
        let out = if normal_item.is_none() & (self.idx < self.insertion_idx) {
            Some(self.filler.clone())
        } else if self.idx == self.insertion_idx {
            self.insertion_item.take()
        } else {
            normal_item
        };
        self.idx += 1;
        out
    }
}


pub trait VectorFileConversions: UninitVectorExpr + ConcreteVectorExpr 
where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    #[cfg(feature = "csv")]
    /// top left corner == (row = 0, col = 0), vector is read top down
    fn read_csv_column<P: AsRef<Path>>(path: P, builder: Self::Builder, row_start: usize, col: usize) -> Result<Self, CSVError<<Self::Output as AsText>::Error>> where Self::Output: AsText {
        let mut csv = csv::Reader::from_path(path)?;
        let mut records = csv.records();
        for _ in 0..row_start {let _ = records.next().ok_or(CSVError::CellOutOfBounds)?;} // don't care if these individual rows are malformed
        let mut uninit = Self::new_uninit(builder);
        let mut num_fields_written = 0;
        let mut error = None;

        // Note: this section *must not return/crash* to avoid leaking
        for _ in 0..uninit.size() {
            let record = match records.next() {
                None => {
                    error = Some(CSVError::CellOutOfBounds);
                    break;
                }
                Some(Err(e)) => {
                    error = Some(CSVError::CSVError(e));
                    break;
                }
                Some(Ok(record)) => record
            };
            if col >= record.len() {
                error = Some(CSVError::CellOutOfBounds);
                break;
            }
            let field = match Self::Output::from_text(&record[col]) {
                Err(e) => {
                    error = Some(CSVError::FieldError(e));
                    break;
                }
                Ok(field) => field
            };
            unsafe { Self::init_index(&mut uninit, num_fields_written, field); }
            num_fields_written += 1;
        }

        if let Some(error) = error {
            unsafe {
                for i in 0..num_fields_written {
                    Self::drop_index(&mut uninit, i);
                }
                Self::drop_ots(&mut uninit);
            }
            return Err(error);
        }
        unsafe { Ok(Self::assume_init(uninit)) }
    }

    #[cfg(feature = "csv")]
    fn write_csv_column<P: AsRef<Path>>(&self, path: P, row_start: usize, col: usize) -> Result<(), CSVError<<Self::Output as AsText>::Error>> 
    where 
        Self::Output: AsText,
    {
        let temp_path = generate_tmp_file_path(&path);
        let mut writer = csv::Writer::from_path(&temp_path)?;
        let mut base_records = if path.as_ref().exists() {
            Some(csv::Reader::from_path(&path)?)
        } else {
            None
        };

        let mut record = csv::StringRecord::new();
        for _ in 0..row_start {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?;
            }
            writer.write_record(&record)?;
        }

        for i in 0..self.size() {
            if let Some(base_records) = &mut base_records {
                base_records.read_record(&mut record)?;
            }
            let field_txt = self[i].as_text();
            writer.write_record(IterInsert::new(record.iter(), &field_txt, col, ""))?;
        }

        writer.flush()?;
        // to make sure the files get closed
        drop(writer);
        drop(base_records);
        
        fs::rename(&temp_path, path)?;
        Ok(())
    }
}