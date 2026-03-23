from dataclasses import dataclass, field
from typing import List, Optional, Any
import os



@dataclass
class Arguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to training data"}
    )
    val_data: str = field(
        default=None, metadata={"help": "Path to validation data"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Path to test data"}
    )

    val_batch_size: int = field(default=32)

    
    teach_device: str = field(default='cuda:1')
    student_device: str = field(default='cuda:0')


    output_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the final model"})
