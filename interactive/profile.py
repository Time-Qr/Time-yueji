# -*- coding: utf-8 -*-
"""可移植的对象工程包：保存偏好、确认过的示例和候选反馈。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


PROFILE_VERSION = 2


@dataclass
class FeedbackRecord:
    image: str
    candidates: list[dict] = field(default_factory=list)
    selected: int | None = None
    worst: int | None = None
    rating: int | None = None
    tags: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ObjectProfile:
    name: str
    description: str = ""
    settings: dict = field(default_factory=dict)
    color_samples_rgb: list[list[int]] = field(default_factory=list)
    feedback: list[FeedbackRecord] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    version: int = PROFILE_VERSION

    def to_dict(self):
        return {
            "format": "interactive-object-profile",
            "version": int(self.version),
            "name": self.name,
            "description": self.description,
            "settings": self.settings,
            "color_samples_rgb": self.color_samples_rgb,
            "examples": self.examples,
            "feedback": [
                {
                    "image": item.image,
                    "candidates": item.candidates,
                    "selected": item.selected,
                    "worst": item.worst,
                    "rating": item.rating,
                    "tags": item.tags,
                    "note": item.note,
                }
                for item in self.feedback
            ],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != "interactive-object-profile":
            raise ValueError("不是交互抠图工程包")
        records = []
        for item in data.get("feedback", []):
            records.append(FeedbackRecord(
                image=str(item.get("image", "")),
                candidates=list(item.get("candidates", [])),
                selected=item.get("selected"),
                worst=item.get("worst"),
                rating=item.get("rating"),
                tags=list(item.get("tags", [])),
                note=str(item.get("note", "")),
            ))
        return cls(
            name=str(data.get("name", "未命名对象")),
            description=str(data.get("description", "")),
            settings=dict(data.get("settings", {})),
            color_samples_rgb=list(data.get("color_samples_rgb", [])),
            feedback=records,
            examples=list(data.get("examples", [])),
            version=int(data.get("version", PROFILE_VERSION)),
        )

    def add_feedback(self, record):
        if not isinstance(record, FeedbackRecord):
            raise TypeError("record 必须是 FeedbackRecord")
        self.feedback.append(record)


def save_profile(profile, path):
    if not isinstance(profile, ObjectProfile):
        raise TypeError("profile 必须是 ObjectProfile")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.to_dict()
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile(path):
    target = Path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    return ObjectProfile.from_dict(data)
