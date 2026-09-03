import pytest
import torch

from e3nn import io, o3
from e3nn.nn import Activation, BatchNorm, Gate, S2Activation
from e3nn.util._argtools import _transform
from e3nn.util.test import assert_equivariant


def test_time_reversal_representation() -> None:
    ir = o3.Irrep("1eo")
    zero = torch.zeros(())

    assert torch.allclose(ir.D_from_angles(zero, zero, zero, kt=torch.ones(())), -torch.eye(3))
    assert torch.allclose(ir.D_from_matrix(-torch.eye(3), parity=False, time_reversal=True), -torch.eye(3))

    scalar = o3.Irrep("0oo")
    assert torch.allclose(scalar.D_from_angles(zero, zero, zero, k=torch.ones(())), -torch.ones(1, 1))
    assert torch.allclose(scalar.D_from_angles(zero, zero, zero, kt=torch.ones(())), -torch.ones(1, 1))
    assert torch.allclose(
        scalar.D_from_angles(zero, zero, zero, k=torch.ones(()), kt=torch.ones(())), torch.ones(1, 1)
    )


def test_time_reversal_transform_helpers() -> None:
    vector = torch.tensor([[1.0, 2.0, 3.0]])
    identity = torch.eye(3)
    irreps = ["cartesian_points", "spin", o3.Irreps("1oe"), o3.Irreps("1eo")]

    parity = _transform([vector] * 4, irreps, identity, parity_k=1)
    time_reversal = _transform([vector] * 4, irreps, identity, time_reversal_k=1)

    for actual, expected in zip(parity, (-vector, vector, -vector, vector)):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(time_reversal, (vector, -vector, vector, -vector)):
        torch.testing.assert_close(actual, expected)

    rotation = o3.rand_matrix(dtype=vector.dtype)
    spin_only = _transform([vector] * 4, irreps, rotation, only_rot_spin=True)
    torch.testing.assert_close(spin_only[0], vector)
    torch.testing.assert_close(spin_only[1], vector @ rotation.T)
    torch.testing.assert_close(spin_only[2], vector)
    torch.testing.assert_close(spin_only[3], vector)


def test_time_reversal_linear() -> None:
    linear = o3.Linear("2x1ee + 3x1eo", "4x1ee + 5x1eo")

    assert {(ins.i_in, ins.i_out) for ins in linear.instructions} == {(0, 0), (1, 1)}
    assert_equivariant(linear, do_time_reversal=True)

    assert o3.Linear("0eo", "0eo", biases=True).bias_numel == 0


def test_time_reversal_batch_norm() -> None:
    batch_norm = BatchNorm("2x0eo").eval()

    assert batch_norm.running_mean.numel() == 0
    assert batch_norm.bias.numel() == 0
    assert_equivariant(batch_norm, irreps_in="2x0eo", irreps_out="2x0eo", do_time_reversal=True)


def test_time_reversal_tensor_product() -> None:
    tensor_product = o3.FullTensorProduct("1eo", "1oe")

    assert tensor_product.irreps_out == o3.Irreps("0oo + 1oo + 2oo")
    assert_equivariant(tensor_product, do_time_reversal=True)


def test_time_reversal_activation() -> None:
    even = Activation("2x0eo", [torch.abs])
    odd = Activation("2x0eo", [torch.tanh])

    assert even.irreps_out == o3.Irreps("2x0ee")
    assert odd.irreps_out == o3.Irreps("2x0eo")
    assert_equivariant(even, do_time_reversal=True)
    assert_equivariant(odd, do_time_reversal=True)

    with pytest.raises(ValueError, match="time-reversal"):
        Activation("2x0eo", [torch.nn.functional.silu])


def test_time_reversal_gate() -> None:
    gate = Gate("2x0eo", [torch.tanh], "3x0eo", [torch.tanh], "3x1ee")

    assert gate.irreps_out == o3.Irreps("2x0eo + 3x1eo")
    assert_equivariant(gate, do_time_reversal=True)


def test_time_reversal_spherical_harmonics() -> None:
    spherical_harmonics = o3.SphericalHarmonics([0, 1, 2, 3], normalize=True, parity=False, time_reversal=True)

    assert spherical_harmonics.irreps_in == o3.Irreps("1eo")
    assert spherical_harmonics.irreps_out == o3.Irreps("0ee + 1eo + 2ee + 3eo")
    assert_equivariant(spherical_harmonics, do_time_reversal=True)

    with pytest.raises(ValueError, match="time-reversal parity"):
        o3.SphericalHarmonics("0ee + 1ee", normalize=True, irreps_in="1eo")


def test_time_reversal_spherical_tensor_helpers() -> None:
    spherical_tensor = io.SphericalTensor(2, p_val=1, p_arg=1, t_val=1, t_arg=-1)
    positions = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.tensor([1.0])

    coefficients = spherical_tensor.sum_of_diracs(positions, values)
    assert coefficients.shape == (spherical_tensor.dim,)
    assert spherical_tensor.signal_xyz(coefficients, positions).shape == (1,)


def test_time_reversal_s2_activation(float_tolerance) -> None:
    spherical_tensor = io.SphericalTensor(2, p_val=1, p_arg=1, t_val=1, t_arg=-1)
    activation = S2Activation(spherical_tensor, torch.tanh, 120)

    assert activation.irreps_out == spherical_tensor
    assert_equivariant(activation, do_time_reversal=True, tolerance=torch.sqrt(float_tolerance))


def test_time_reversal_experimental_tensor_products() -> None:
    irreps_in1 = o3.Irreps("1eo + 1ee")
    irreps_in2 = o3.Irreps("1oe + 1oe")
    input1 = irreps_in1.randn(3, -1)
    input2 = irreps_in2.randn(3, -1)

    reference_full = o3.FullTensorProduct(irreps_in1, "1oe")
    experimental_full = o3.experimental.FullTensorProductv2(irreps_in1, "1oe")
    assert experimental_full.irreps_out == reference_full.irreps_out
    torch.testing.assert_close(experimental_full(input1, input2[..., :3]), reference_full(input1, input2[..., :3]))

    reference_elementwise = o3.ElementwiseTensorProduct(irreps_in1, irreps_in2)
    experimental_elementwise = o3.experimental.ElementwiseTensorProductv2(irreps_in1, irreps_in2)
    reference_output = reference_elementwise(input1, input2)
    experimental_output = experimental_elementwise(input1, input2)
    reference_by_irrep = {
        mul_ir.ir: reference_output[..., ir_slice]
        for mul_ir, ir_slice in zip(reference_elementwise.irreps_out, reference_elementwise.irreps_out.slices())
    }
    for mul_ir, ir_slice in zip(experimental_elementwise.irreps_out, experimental_elementwise.irreps_out.slices()):
        torch.testing.assert_close(experimental_output[..., ir_slice], reference_by_irrep[mul_ir.ir])
