//! Deserialize the live Python scene description without a JSON string or
//! an intermediate serde_json::Value tree. Large arrays use PyBuffer separately.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::de::{self, DeserializeSeed, IntoDeserializer, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

type Error = de::value::Error;

struct Value<'py>(Bound<'py, PyAny>, usize);

fn error(e: impl std::fmt::Display) -> Error {
    de::Error::custom(e)
}

pub fn from_python<T: for<'de> Deserialize<'de>>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    T::deserialize(Value(value.clone(), 0))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("scene: {e}")))
}

impl<'de> de::Deserializer<'de> for Value<'_> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        if self.1 > 128 {
            return Err(error("scene nesting exceeds 128 levels (possible cycle)"));
        }
        let depth = self.1 + 1;
        let value = self.0;
        if value.is_none() {
            visitor.visit_unit()
        } else if value.is_instance_of::<PyBool>() {
            visitor.visit_bool(value.extract().map_err(error)?)
        } else if value.is_instance_of::<PyInt>() {
            match value.extract::<i64>() {
                Ok(n) => visitor.visit_i64(n),
                Err(_) => visitor.visit_u64(value.extract().map_err(error)?),
            }
        } else if value.is_instance_of::<PyFloat>() {
            let number: f64 = value.extract().map_err(error)?;
            if !number.is_finite() {
                return Err(error("non-finite scene number"));
            }
            visitor.visit_f64(number)
        } else if let Ok(s) = value.downcast::<PyString>() {
            visitor.visit_str(s.to_str().map_err(error)?)
        } else if let Ok(dict) = value.downcast::<PyDict>() {
            visitor.visit_map(Dict {
                iter: dict.iter(),
                pending: None,
                depth,
            })
        } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            let len = value.len().map_err(error)?;
            visitor.visit_seq(Sequence {
                value,
                index: 0,
                len,
                depth,
            })
        } else {
            Err(error(format!(
                "unsupported Python value {}",
                value.get_type().name().map_err(error)?
            )))
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        if self.0.is_none() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        let name: String = self.0.extract().map_err(error)?;
        visitor.visit_enum(name.into_deserializer())
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf
        unit unit_struct seq tuple tuple_struct map struct identifier
    }
}

struct Sequence<'py> {
    value: Bound<'py, PyAny>,
    index: usize,
    len: usize,
    depth: usize,
}

impl<'de> SeqAccess<'de> for Sequence<'_> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Error> {
        if self.index == self.len {
            return Ok(None);
        }
        let value = self.value.get_item(self.index).map_err(error)?;
        self.index += 1;
        seed.deserialize(Value(value, self.depth)).map(Some)
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.len - self.index)
    }
}

struct Dict<'py> {
    iter: pyo3::types::iter::BoundDictIterator<'py>,
    pending: Option<Bound<'py, PyAny>>,
    depth: usize,
}

impl<'de> MapAccess<'de> for Dict<'_> {
    type Error = Error;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Error> {
        match self.iter.next() {
            Some((key, value)) => {
                self.pending = Some(value);
                seed.deserialize(Value(key, self.depth)).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(&mut self, seed: V) -> Result<V::Value, Error> {
        let value = self
            .pending
            .take()
            .ok_or_else(|| error("missing map value"))?;
        seed.deserialize(Value(value, self.depth))
    }
}
