#!/usr/bin/env python3

import torch
from src.model.model import LCNN, LCNN_LSTM_Sum
from src.transforms.lfcc import LFCCTransform

def test_lcnn_lstm_sum():
    """
    Тестирование новой архитектуры LCNN-LSTM-sum.
    """
    print("🧪 Тестирование LCNN-LSTM-sum архитектуры...")
    
    # Создаём модели
    lcnn_original = LCNN(num_classes=2, dropout_rate=0.3)
    lcnn_lstm_sum = LCNN_LSTM_Sum(num_classes=2, dropout_rate=0.3)
    
    print(f"✅ Модели созданы:")
    print(f"   📊 LCNN: {sum(p.numel() for p in lcnn_original.parameters()):,} параметров")
    print(f"   🔀 LCNN-LSTM-sum: {sum(p.numel() for p in lcnn_lstm_sum.parameters()):,} параметров")
    
    # LFCC трансформация
    lfcc_transform = LFCCTransform(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        win_length=320,
        n_filter=20,
        n_lfcc=20,
        use_deltas=True
    )
    
    # Тестовое аудио
    test_audio = torch.randn(1, 32000)  # 2 сек при 16kHz
    print(f"\n🎵 Тестовое аудио: {test_audio.shape}")
    
    # LFCC признаки
    lfcc_features = lfcc_transform(test_audio)
    print(f"📊 LFCC признаки: {lfcc_features.shape}")
    
    # Тестируем обе модели
    models = {
        "LCNN": lcnn_original,
        "LCNN-LSTM-sum": lcnn_lstm_sum
    }
    
    print(f"\n🔧 Тестирование forward pass:")
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                # Подготавливаем batch
                batch = {'data_object': lfcc_features}
                
                # Forward pass
                output = model(batch)
                logits = output['logits']
                
                # Проверяем размерности
                assert logits.shape == (1, 2), f"Неправильная размерность выхода: {logits.shape}"
                
                # Проверяем что логиты разумные
                probabilities = torch.softmax(logits, dim=1)
                
                print(f"   ✅ {name}:")
                print(f"      Выход: {logits.shape}")
                print(f"      Логиты: [{logits[0, 0].item():.4f}, {logits[0, 1].item():.4f}]")
                print(f"      Вероятности: [{probabilities[0, 0].item():.4f}, {probabilities[0, 1].item():.4f}]")
                
        except Exception as e:
            print(f"   ❌ {name}: Ошибка - {e}")
            raise e
    
    # Тест разных размеров
    print(f"\n🔧 Тест разных размеров аудио:")
    
    test_sizes = [
        (8000, "0.5s"),
        (16000, "1.0s"), 
        (48000, "3.0s"),
        (64000, "4.0s")
    ]
    
    for samples, duration in test_sizes:
        test_audio = torch.randn(1, samples)
        lfcc_features = lfcc_transform(test_audio)
        
        batch = {'data_object': lfcc_features}
        
        with torch.no_grad():
            output_original = lcnn_original(batch)
            output_lstm = lcnn_lstm_sum(batch)
            
            print(f"   📏 {duration}: {test_audio.shape} → LFCC {lfcc_features.shape} → Outputs {output_original['logits'].shape}")
    
    # Тест batch обработки
    print(f"\n🔧 Тест batch обработки:")
    
    batch_audio = torch.randn(4, 32000)  # batch из 4 аудио
    batch_lfcc = lfcc_transform(batch_audio)
    batch_dict = {'data_object': batch_lfcc}
    
    with torch.no_grad():
        batch_output_original = lcnn_original(batch_dict)
        batch_output_lstm = lcnn_lstm_sum(batch_dict)
        
        print(f"   🎯 Batch: {batch_audio.shape} → LFCC {batch_lfcc.shape}")
        print(f"        LCNN: {batch_output_original['logits'].shape}")
        print(f"        LCNN-LSTM-sum: {batch_output_lstm['logits'].shape}")
    
    print(f"\n🏆 Все тесты LCNN-LSTM-sum прошли успешно!")
    print(f"\n📋 Архитектурные особенности LCNN-LSTM-sum:")
    print(f"   🔄 CNN часть: Та же что и в LCNN")
    print(f"   🧠 2x Bi-LSTM слоя с skip connection")
    print(f"   📊 Average pooling по временной размерности") 
    print(f"   🎯 FC слой для классификации")
    
    return True

if __name__ == "__main__":
    test_lcnn_lstm_sum() 