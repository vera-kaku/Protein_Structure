import glob, os
import tensorflow as tf
import numpy as np
from tensorflow.io import FixedLenFeature, FixedLenSequenceFeature

NUM_RESIDUES = 256
NUM_EXTRA_SEQ = 10
NUM_DIMENSIONS = 3
NUM_AMINO_ACIDS = 21
# Abbreviation	1 letter abbreviation	Amino acid name
# twenty amino acids (that make up proteins) + 1 for representing a gap in an alignment
AMINO_ACIDS = [
    record.split()
    for record in """Ala	A	Alanine
Arg	R	Arginine
Asn	N	Asparagine
Asp	D	Aspartic acid
Cys	C	Cysteine
Gln	Q	Glutamine
Glu	E	Glutamic acid
Gly	G	Glycine
His	H	Histidine
Ile	I	Isoleucine
Leu	L	Leucine
Lys	K	Lysine
Met	M	Methionine
Phe	F	Phenylalanine
Pro	P	Proline
Ser	S	Serine
Thr	T	Threonine
Trp	W	Tryptophan
Tyr	Y	Tyrosine
Val	V	Valine
GAP _   Alignment gap""".splitlines()
]
# unused codes
# Pyl	O	Pyrrolysine
# Sec	U	Selenocysteine
# Asx	B	Aspartic acid or Asparagine
# Glx	Z	Glutamic acid or Glutamine
# Xaa	X	Any amino acid
# Xle	J	Leucine or Isoleucine
AMINO_ACID_MAP = {r[1]: r[2] for r in AMINO_ACIDS}
AMINO_ACID_BY_INDEX = list(
    sorted(AMINO_ACID_MAP.keys())
)  # 0-20 inclusive, protein net numbering
AMINO_ACID_NUMBER = {key: number for number, key in enumerate(AMINO_ACID_BY_INDEX)}
AMINO_ACID_GAP_INDEX = AMINO_ACID_BY_INDEX.index("_")


def to_amino_acid_string(amino_acid_indices):
    return "".join([AMINO_ACID_BY_INDEX[x] for x in amino_acid_indices])


def to_amino_acid_numerical(amino_acid_codes):
    return np.array(
        [
            AMINO_ACID_NUMBER[x] if x != "-" else AMINO_ACID_GAP_INDEX
            for x in amino_acid_codes
        ],
        dtype=np.int64,
    )


ID_MAXLEN = 16


def rmse(predicted_points, true_points, true_points_mask, per_residue=False):
    """
    Root mean squared error of euclidean distance between predicted and true points

    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        per_residue: If true, return score for each residue.

    Returns:
        RMSE
    """
    if len(true_points_mask.shape) == 2:
        true_points_mask = tf.expand_dims(true_points_mask, axis=2)
    reduce_axes = (-1,) if per_residue else (-2, -1)
    rmse = tf.sqrt(
        1e-10
        + tf.reduce_sum(
            ((predicted_points - true_points) ** 2) * true_points_mask, axis=reduce_axes
        )
    )
    if any(rmse == np.nan):
        print(predicted_points, true_points, true_points_mask)
    return rmse


# based on Google's Alphafold implementation:
# https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/lddt.py
# adapted for use in TensorFlow
def lddt(
    predicted_points,
    true_points,
    true_points_mask,
    cutoff=15.0,
    per_residue=False,
    differentiable=False,
):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
    # `predicted_points` is a tensor that represents the predicted 3D coordinates of a protein
    # structure. It has shape (batch, length, 3), where `batch` is the number of protein structures in
    # the batch, `length` is the number of residues in each protein structure, and `3` represents the
    # x, y, and z coordinates of each residue.
    predicted_points: (batch, length, 3) array of predicted 3D points
    true_points: (batch, length, 3) array of true 3D points
    # The `true_points_mask` is a binary-valued float array that indicates which points in the true
    # points exist. It has a shape of `(batch, length, 1)`, where `batch` is the number of protein
    # structures in the batch and `length` is the number of residues in each protein structure. The
    # mask should have a value of 1 for points that exist in the true points and 0 for points that do
    # not exist. This mask is used in calculations such as RMSE and lDDT to exclude points that do not
    # exist in the true points from the calculations.
    # true_points_mask: (batch, length, 1) binary-valued float array.  This mask
    # should be 1 for points that exist in the true points.
    # cutoff: Maximum distance for a pair of points to be included
    # per_residue: If true, return score for each residue.  Note that the overall
    # lDDT is not exactly the mean of the per_residue lDDT's because some
    # residues have more contacts than others.

    Returns:
    An (approximate, see above) lDDT score in the range 0-1.
    """
    # print(predicted_points.shape, true_points.shape, true_points_mask.shape)
    # assert len(predicted_points.shape) == 3
    # assert predicted_points.shape[-1] == 3
    if len(true_points_mask.shape) == 2:
        true_points_mask = tf.expand_dims(true_points_mask, axis=2)
    # assert true_points_mask.shape[-1] == 1
    # assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = tf.sqrt(
        1e-10
        + tf.reduce_sum(
            (true_points[:, :, None] - true_points[:, None, :]) ** 2, axis=-1
        )
    )

    dmat_predicted = tf.sqrt(
        1e-10
        + tf.reduce_sum(
            (predicted_points[:, :, None] - predicted_points[:, None, :]) ** 2, axis=-1
        )
    )

    dists_to_score = (
        tf.cast(dmat_true < cutoff, tf.float32)
        * true_points_mask
        * tf.transpose(true_points_mask, [0, 2, 1])
        * (1.0 - tf.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = tf.abs(dmat_true - dmat_predicted)

    if differentiable:
        # original would score sum(1,0.75,0.5,0.25 for each value in range [< 0.5, <1, <2, <4])
        # close approximation, would be 1 at 0 and 0 at 4
        score = 3 - tf.math.log(4 + dist_l1) / tf.math.log(2.0)
    else:
        # True lDDT uses a number of fixed bins.
        # We ignore the physical plausibility correction to lDDT, though.
        score = 0.25 * (
            tf.cast(dist_l1 < 0.5, tf.float32)
            + tf.cast(dist_l1 < 1.0, tf.float32)
            + tf.cast(dist_l1 < 2.0, tf.float32)
            + tf.cast(dist_l1 < 4.0, tf.float32)
        )

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (1e-10 + tf.reduce_sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + tf.reduce_sum(dists_to_score * score, axis=reduce_axes))
    return score


def decode_fn_with_msainfo(example_proto, debug=False):
    feature_description = {
        "id": tf.io.FixedLenSequenceFeature(
            [], tf.string, default_value="", allow_missing=True
        ),
        "primary": tf.io.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True
        ),
        "tertiary": tf.io.FixedLenSequenceFeature(
            [NUM_DIMENSIONS], tf.float32, default_value=0, allow_missing=True
        ),
        "mask": tf.io.FixedLenSequenceFeature(
            [], tf.float32, default_value=0, allow_missing=True
        ),
        "evolutionary": tf.io.FixedLenSequenceFeature(
            [], tf.float32, default_value=0, allow_missing=True
        ),
        "msa": tf.io.FixedLenSequenceFeature(
            [NUM_RESIDUES], tf.int64, default_value=0, allow_missing=True
        ),
        "msa_score": tf.io.FixedLenSequenceFeature(
            [], tf.float32, default_value=0, allow_missing=True
        ),
        "extra_structure": tf.io.FixedLenSequenceFeature(
            [NUM_RESIDUES, NUM_DIMENSIONS],
            tf.float32,
            default_value=0,
            allow_missing=True,
        ),
        "extra_mask": tf.io.FixedLenSequenceFeature(
            [NUM_RESIDUES], tf.float32, default_value=0, allow_missing=True
        ),
    }
    record = tf.io.parse_single_example(example_proto, feature_description)
    record["primary_onehot"] = tf.one_hot(record["primary"], NUM_AMINO_ACIDS)
    return record


def decode_fn(record_bytes, num_evo_entries=21):
    record = tf.io.parse_single_sequence_example(
        record_bytes,
        context_features={"id": FixedLenFeature((1,), tf.string)},
        sequence_features={
            "primary": FixedLenSequenceFeature((1,), tf.int64),
            "evolutionary": FixedLenSequenceFeature(
                (num_evo_entries,), tf.float32, allow_missing=True
            ), 
            "secondary": FixedLenSequenceFeature((1,), tf.int64, allow_missing=True),
            "tertiary": FixedLenSequenceFeature(
                (NUM_DIMENSIONS,), tf.float32, allow_missing=True
            ),
            "mask": FixedLenSequenceFeature((1,), tf.float32, allow_missing=True),
        },
    )
    # scale the picometers to pdb format
    record[1]["tertiary"] /= 100
    record[1]["primary_onehot"] = tf.one_hot(record[1]["primary"], NUM_AMINO_ACIDS)
    return record


def load_data_with_msa(data_folder, filename):
    file_path = data_folder + filename
    if not os.path.exists(file_path):
        raise (ValueError(f"no data found in: {file_path}"))

    raw_dataset = tf.data.TFRecordDataset(file_path)
    decoded_dataset = raw_dataset.map(decode_fn_with_msainfo)
    # debug info
    # print("Debug info, shape of one data record:")
    # for record in decoded_dataset.take(1):
    #     for key, value in record.items():
    #         print(key,"shape=",value.shape)
    return decoded_dataset


def load_data(data_folder, mode="training/50", competition="new_data"):
    data_files = []
    for folder, _, files in os.walk(data_folder + competition + "/" + mode + "/"):
        data_files.extend(
            [folder + "/" + file for file in files if not file.startswith(".")]
        )
    if not data_files:
        raise (ValueError(f"no data found in folder: {data_folder}"))
    return tf.data.TFRecordDataset(data_files).map(decode_fn)
