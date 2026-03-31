//! Fast binary PLY mesh loader.

use std::io::Read;
use std::path::Path;

pub struct PlyMesh {
    pub vertices: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<i32>,
}

#[derive(Clone, Copy)]
enum PropType {
    Float,
    Double,
    Uchar,
    Int,
    UInt,
    Short,
    UShort,
}

#[derive(Clone)]
enum ElementProp {
    Scalar(PropType),
    List {
        count_type: PropType,
        val_type: PropType,
    },
}

struct ElementDef {
    name: String,
    count: usize,
    props: Vec<(String, ElementProp)>,
}

pub fn load(path: &Path) -> Option<PlyMesh> {
    use std::io::BufRead;

    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open PLY file {}: {e}", path.display()));

    let buf_file = std::io::BufReader::new(file);
    let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz")
        || path.to_string_lossy().contains(".ply.gz")
    {
        Box::new(flate2::read::GzDecoder::new(buf_file))
    } else {
        Box::new(buf_file)
    };
    let mut reader = std::io::BufReader::new(reader);

    // Parse header
    let mut is_binary_le = false;
    let mut elements: Vec<ElementDef> = Vec::new();

    loop {
        let mut line = String::new();
        if reader
            .read_line(&mut line)
            .expect("Failed to read PLY header")
            == 0
        {
            eprintln!("Unexpected EOF in PLY header: {}", path.display());
            return None;
        }
        let line = line.trim();
        if line == "end_header" {
            break;
        }
        if line.is_empty() || line.starts_with("comment") || line.starts_with("obj_info") {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "ply" => {}
            "format" => {
                is_binary_le = parts.get(1) == Some(&"binary_little_endian");
                if !is_binary_le {
                    eprintln!(
                        "Only binary_little_endian PLY is supported (got {:?}): {}",
                        parts.get(1),
                        path.display()
                    );
                    return None;
                }
            }
            "element" if parts.len() >= 3 => {
                elements.push(ElementDef {
                    name: parts[1].to_string(),
                    count: parts[2].parse().unwrap_or(0),
                    props: Vec::new(),
                });
            }
            "property" if !elements.is_empty() => {
                let elem = elements.last_mut().unwrap();
                if parts.len() >= 3 && parts[1] == "list" && parts.len() >= 5 {
                    elem.props.push((
                        parts[4].to_string(),
                        ElementProp::List {
                            count_type: parse_prop_type(parts[2]),
                            val_type: parse_prop_type(parts[3]),
                        },
                    ));
                } else if parts.len() >= 3 {
                    elem.props.push((
                        parts[2].to_string(),
                        ElementProp::Scalar(parse_prop_type(parts[1])),
                    ));
                }
            }
            _ => {}
        }
    }

    if !is_binary_le {
        return None;
    }

    // Find vertex and face elements
    let vertex_elem = elements.iter().find(|e| e.name == "vertex");
    let face_elem = elements.iter().find(|e| e.name == "face");

    let num_vertices = vertex_elem.map_or(0, |e| e.count);
    let num_faces = face_elem.map_or(0, |e| e.count);

    // Determine vertex property roles
    #[derive(Clone, Copy, PartialEq)]
    enum PropRole {
        X,
        Y,
        Z,
        Nx,
        Ny,
        Nz,
        U,
        V,
        Other,
    }

    let vertex_props: Vec<(PropType, PropRole)> = vertex_elem
        .map(|e| {
            e.props
                .iter()
                .filter_map(|(name, prop)| {
                    if let ElementProp::Scalar(ptype) = prop {
                        let role = match name.as_str() {
                            "x" => PropRole::X,
                            "y" => PropRole::Y,
                            "z" => PropRole::Z,
                            "nx" => PropRole::Nx,
                            "ny" => PropRole::Ny,
                            "nz" => PropRole::Nz,
                            "u" | "s" | "texture_u" => PropRole::U,
                            "v" | "t" | "texture_v" => PropRole::V,
                            _ => PropRole::Other,
                        };
                        Some((*ptype, role))
                    } else {
                        None // skip list properties on vertices
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    // Find the face index list property
    let face_index_prop = face_elem.and_then(|e| {
        e.props.iter().find_map(|(name, prop)| {
            if let ElementProp::List {
                count_type,
                val_type,
            } = prop
            {
                if name == "vertex_indices" || name == "vertex_index" {
                    return Some((*count_type, *val_type));
                }
            }
            None
        })
    });

    // Count non-index face properties to skip
    let face_extra_props: Vec<ElementProp> = face_elem
        .map(|e| {
            e.props
                .iter()
                .filter(|(name, _)| name != "vertex_indices" && name != "vertex_index")
                .map(|(_, prop)| prop.clone())
                .collect()
        })
        .unwrap_or_default();

    let has_normals = vertex_props.iter().any(|(_, r)| *r == PropRole::Nx);
    let has_uvs = vertex_props.iter().any(|(_, r)| *r == PropRole::U);

    // Read all elements in order
    let mut vertices = Vec::with_capacity(num_vertices * 3);
    let mut normals = if has_normals {
        Vec::with_capacity(num_vertices * 3)
    } else {
        Vec::new()
    };
    let mut texcoords = if has_uvs {
        Vec::with_capacity(num_vertices * 2)
    } else {
        Vec::new()
    };
    let mut indices = Vec::with_capacity(num_faces * 3);

    for elem in &elements {
        if elem.name == "vertex" {
            for _ in 0..elem.count {
                for &(ptype, role) in &vertex_props {
                    let val = read_prop_f32(&mut reader, ptype);
                    match role {
                        PropRole::X | PropRole::Y | PropRole::Z => vertices.push(val),
                        PropRole::Nx | PropRole::Ny | PropRole::Nz => normals.push(val),
                        PropRole::U | PropRole::V => texcoords.push(val),
                        PropRole::Other => {}
                    }
                }
            }
        } else if elem.name == "face" {
            let (face_count_type, face_val_type) =
                face_index_prop.unwrap_or((PropType::Uchar, PropType::Int));
            for _ in 0..elem.count {
                let count = read_prop_f32(&mut reader, face_count_type) as usize;
                let mut face_indices = Vec::with_capacity(count);
                for _ in 0..count {
                    face_indices.push(read_prop_f32(&mut reader, face_val_type) as i32);
                }
                // Triangulate (fan)
                if count >= 3 {
                    for i in 1..count - 1 {
                        indices.push(face_indices[0]);
                        indices.push(face_indices[i]);
                        indices.push(face_indices[i + 1]);
                    }
                }
                // Skip extra face properties
                for prop in &face_extra_props {
                    match prop {
                        ElementProp::Scalar(ptype) => {
                            read_prop_f32(&mut reader, *ptype);
                        }
                        ElementProp::List {
                            count_type,
                            val_type,
                        } => {
                            let n = read_prop_f32(&mut reader, *count_type) as usize;
                            for _ in 0..n {
                                read_prop_f32(&mut reader, *val_type);
                            }
                        }
                    }
                }
            }
        } else {
            // Skip unknown elements
            for _ in 0..elem.count {
                for (_, prop) in &elem.props {
                    match prop {
                        ElementProp::Scalar(ptype) => {
                            read_prop_f32(&mut reader, *ptype);
                        }
                        ElementProp::List {
                            count_type,
                            val_type,
                        } => {
                            let n = read_prop_f32(&mut reader, *count_type) as usize;
                            for _ in 0..n {
                                read_prop_f32(&mut reader, *val_type);
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "Loaded PLY: {} vertices, {} triangles from {}",
        vertices.len() / 3,
        indices.len() / 3,
        path.display()
    );

    Some(PlyMesh {
        vertices,
        normals,
        texcoords,
        indices,
    })
}

fn parse_prop_type(s: &str) -> PropType {
    match s {
        "float" | "float32" => PropType::Float,
        "double" | "float64" => PropType::Double,
        "uchar" | "uint8" => PropType::Uchar,
        "int" | "int32" => PropType::Int,
        "uint" | "uint32" => PropType::UInt,
        "short" | "int16" => PropType::Short,
        "ushort" | "uint16" => PropType::UShort,
        _ => {
            eprintln!("  warning: unknown PLY property type: {s}, assuming float");
            PropType::Float
        }
    }
}

fn read_prop_f32(r: &mut impl Read, ptype: PropType) -> f32 {
    match ptype {
        PropType::Float => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            f32::from_le_bytes(b)
        }
        PropType::Double => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b).unwrap();
            f64::from_le_bytes(b) as f32
        }
        PropType::Uchar => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).unwrap();
            b[0] as f32
        }
        PropType::Int => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            i32::from_le_bytes(b) as f32
        }
        PropType::UInt => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).unwrap();
            u32::from_le_bytes(b) as f32
        }
        PropType::Short => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).unwrap();
            i16::from_le_bytes(b) as f32
        }
        PropType::UShort => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).unwrap();
            u16::from_le_bytes(b) as f32
        }
    }
}
