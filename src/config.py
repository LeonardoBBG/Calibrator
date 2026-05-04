from dataclasses import dataclass
from pathlib import Path
from typing import List

def default_require_temperature_support(model_name: str) -> bool:
    """GPT-5 family models may require provider-default temperature."""
    normalized = model_name.strip().lower()
    return not normalized.startswith("gpt-5")

@dataclass
class Config:
    project_root: Path
    ws_path: Path
    judgment_path: Path
    judgments_dir: Path
    run_mode: str
    dictionary_path: Path
    ws_tagging_prompt_path: Path
    calibration_prompt_path: Path
    compression_prompt_path: Path
    repair_prompt_path: Path
    outcome_optimization_prompt_path: Path
    outcome_repair_prompt_path: Path
    output_root: Path
    cache_root: Path
    model_name: str
    api_provider: str
    temperature: float
    require_temperature_support: bool
    max_tokens: int
    max_repair_attempts: int
    max_outcome_repair_attempts: int
    cache_enabled: bool
    validate_json_writes: bool
    ocr_enabled: bool
    debug: bool
    run_id: str

    def selected_judgment_paths(self) -> List[Path]:
        """Return judgment PDFs for the configured run mode."""
        if self.run_mode == "debug":
            return [self.judgment_path]
        if self.run_mode == "batch":
            return sorted(self.judgments_dir.glob("*.pdf"))
        raise ValueError("run_mode must be 'debug' or 'batch'")

    @classmethod
    def default(cls, run_id: str) -> 'Config':
        project_root = Path("/home/hello/Projects/Calibrator")
        return cls(
            project_root=project_root,
            ws_path=project_root / "input" / "ws" / "witness_statement.pdf",
            judgment_path=project_root / "input" / "judgments" / "Mr_B_Burke_v_Thomas_Contracting_Ltd_and_Thomas_Plant_Hire_Ltd_-_2414977_2018_-_Reserved.pdf",
            judgments_dir=project_root / "input" / "judgments",
            run_mode="debug",
            dictionary_path=project_root / "input" / "dictionary" / "WS_Controlled_Theme_Dictionary_v1_2_final.json",
            ws_tagging_prompt_path=project_root / "input" / "prompts" / "ws_tagging_prompt.txt",
            calibration_prompt_path=project_root / "input" / "prompts" / "calibration_prompt.txt",
            compression_prompt_path=project_root / "input" / "prompts" / "compression_prompt.txt",
            repair_prompt_path=project_root / "input" / "prompts" / "repair_prompt.txt",
            outcome_optimization_prompt_path=project_root / "input" / "prompts" / "outcome_optimization_prompt.txt",
            outcome_repair_prompt_path=project_root / "input" / "prompts" / "outcome_repair_prompt.txt",
            output_root=project_root / "output",
            cache_root=project_root / "output" / "cache",
            model_name="gpt-4.1-mini",
            api_provider="openai",
            temperature=0.0,
            require_temperature_support=True,
            max_tokens=12000,
            max_repair_attempts=3,
            max_outcome_repair_attempts=1,
            cache_enabled=True,
            validate_json_writes=False,
            ocr_enabled=False,
            debug=False,
            run_id=run_id
        )
