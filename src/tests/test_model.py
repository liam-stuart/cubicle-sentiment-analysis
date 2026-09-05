import torch
import pytest
import torch.nn as nn
from model import Model


def test_models_work():
    test_data = torch.randint(0, 9, (50, 50))
    model_dict = {"GRU": nn.GRU,
                  "LSTM": nn.LSTM,
                  "RNN": nn.RNN}

    for model_name in model_dict:
        model = Model(model_name, 11, 32, 32)
        assert any(isinstance(m, nn.Embedding) for m in model.modules())
        assert any(isinstance(m, model_dict[model_name]) for m in model.modules())
        assert any(isinstance(m, nn.Linear) for m in model.modules())

        with torch.no_grad():
            output = model(test_data)
        assert output.shape == torch.Size([50, 1])


def test_invalid_model_name():
    with pytest.raises(ValueError):
        Model("XGBoost", 11, 32, 32)
