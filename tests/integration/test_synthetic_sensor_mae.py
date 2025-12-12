import torch

from alchemi.data.datasets import SyntheticSensorDataset
from alchemi.spectral import Sample, Spectrum
from alchemi.srf.synthetic import SyntheticSensorConfig
from alchemi.models import AnySensorIngest
from alchemi.models.backbone.mae import MAEBackbone
from alchemi.tokens.band_tokenizer import BandTokenizer
from alchemi.types import QuantityKind


def test_synthetic_sensor_pipeline_supports_ingest_and_mae_training():
    axis = torch.linspace(400.0, 1000.0, 600).numpy()
    lab_values = (torch.sin(torch.from_numpy(axis) / 150.0) * 0.25 + 0.5).numpy()
    lab_sample = Sample(
        spectrum=Spectrum(wavelength_nm=axis, values=lab_values, kind=QuantityKind.REFLECTANCE),
        sensor_id="lab",
    )

    synth_cfg = SyntheticSensorConfig(
        highres_axis_nm=axis,
        n_bands=24,
        center_jitter_nm=1.0,
        fwhm_range_nm=(6.0, 12.0),
        shape="gaussian",
        seed=42,
    )
    dataset = SyntheticSensorDataset([lab_sample], synth_cfg)
    item = dataset[0]
    sample = item["sample"]

    tokenizer = BandTokenizer()
    _ = tokenizer(
        sample.spectrum.values,
        sample.spectrum.wavelength_nm,
        axis_unit="nm",
        width=sample.band_meta.width_nm,
        width_from_default=sample.band_meta.width_from_default,
        srf_row=sample.srf_matrix.matrix,
        srf_provenance=sample.band_meta.srf_provenance,
    )

    ingest = AnySensorIngest(d_model=64, group_size=1, patch_size=1, max_sensors=2)
    backbone = MAEBackbone(embed_dim=64, depth=1, num_heads=4, masking_ratio=0.25, decoder_dim=32, decoder_depth=1)
    opt = torch.optim.SGD(backbone.parameters(), lr=0.01)

    for _ in range(2):
        ingest_out = ingest(sample)
        tokens = ingest_out.tokens.detach()
        out = backbone.forward_mae(tokens)
        loss = out.decoded.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

    assert ingest_out.tokens.shape[-1] == backbone.embed_dim
    assert out.decoded.shape[-1] == backbone.embed_dim
