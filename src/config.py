from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    project_root: Path
    ws_path: Path
    judgment_path: Path
    dictionary_path: Path
    ws_tagging_prompt_path: Path
    calibration_prompt_path: Path
    compression_prompt_path: Path
    repair_prompt_path: Path
    output_root: Path
    cache_root: Path
    model_name: str
    api_provider: str
    temperature: float
    max_tokens: int
    max_repair_attempts: int
    cache_enabled: bool
    validate_json_writes: bool
    ocr_enabled: bool
    debug: bool
    run_id: str

    @classmethod
    def default(cls, run_id: str) -> 'Config':
        project_root = Path("/home/hello/Projects/Calibrator")
        return cls(
            project_root=project_root,
            ws_path=project_root / "input" / "ws" / "witness_statement.pdf",
            judgment_path=project_root / "input" / "judgments" / "Mr_B_Burke_v_Thomas_Contracting_Ltd_and_Thomas_Plant_Hire_Ltd_-_2414977_2018_-_Reserved.pdf",
            dictionary_path=project_root / "input" / "dictionary" / "WS_Controlled_Theme_Dictionary_v1_2_final.json",
            ws_tagging_prompt_path=project_root / "input" / "prompts" / "prompt",
            calibration_prompt_path=project_root / "input" / "prompts" / "calibration_prompt.txt",
            compression_prompt_path=project_root / "input" / "prompts" / "compression_prompt.txt",
            repair_prompt_path=project_root / "input" / "prompts" / "repair_prompt.txt",
            output_root=project_root / "output",
            cache_root=project_root / "output" / "cache",
            model_name="gpt-4.1-mini",
            api_provider="openai",
            temperature=0.0,
            max_tokens=12000,
            max_repair_attempts=3,
            cache_enabled=True,
            validate_json_writes=False,
            ocr_enabled=False,
            debug=False,
            run_id=run_id
        )
