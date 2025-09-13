import torch
from hyperpyyaml import load_hyperpyyaml


if __name__ == "__main__":
    pretrain_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/acoustics/qwen/CosyVoice-BlankEN"
    vc_model_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/llm_v2.pt"
    vc_config_path = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/vc_config_v2.3.yaml"
    save_root = "/data/megastore/SHARE/TTS/LAM_TTS/latest/checkpoints/LAM-VC/LLM/lora/MultiEmotion"

    state_dict = torch.load(vc_model_path)
    print(state_dict.keys())
    with open(vc_config_path, 'r') as f:
        vc_configs = load_hyperpyyaml(f, overrides={
            'qwen_pretrain_path': pretrain_path,
            'qwen_sglang_config': None,
            'flow': None,  # llm加载时不需要加载flow和hift模块
            'hift': None,
        })

    VC_model = vc_configs['llm']
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=vc_configs['llm_lora_r'],
        lora_alpha=vc_configs['llm_lora_alpha'],
        target_modules=vc_configs['llm_lora_target_modules'],
        modules_to_save=None,
    )
    VC_model = get_peft_model(VC_model, peft_config)
    VC_model.print_trainable_parameters()

    load_info = VC_model.load_state_dict(state_dict, strict=False)
    print(load_info)
    VC_model.save_pretrained(save_root)