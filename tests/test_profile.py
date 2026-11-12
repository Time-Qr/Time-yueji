from __future__ import annotations

from interactive.profile import (
    FeedbackRecord,
    ObjectProfile,
    load_profile,
    save_profile,
)


def test_profile_round_trip(tmp_path):
    profile = ObjectProfile(
        "绿萝",
        description="保留枝叶，排除花盆",
        settings={"color_assist_strength": 0.25},
        color_samples_rgb=[[20, 150, 30]],
        examples=[{
            "name": "plant-01",
            "image": "plant.png",
            "points": [{"x": 3.0, "y": 4.0, "label": 1}],
        }],
    )
    profile.add_feedback(FeedbackRecord(
        image="plant.png",
        candidates=[{"index": 0, "sam_score": 0.9}],
        selected=0,
        rating=5,
        tags=["漏细枝"],
        note="保留最好",
    ))
    path = tmp_path / "绿萝.json"
    save_profile(profile, path)
    loaded = load_profile(path)
    assert loaded.name == "绿萝"
    assert loaded.settings["color_assist_strength"] == 0.25
    assert loaded.feedback[0].tags == ["漏细枝"]
    assert loaded.examples[0]["name"] == "plant-01"
