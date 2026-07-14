"""Preprocess Amazon MoviesAndTV user-item data into a user-centric JSON file.

Reads:
- datasets_orig/Amazon/MoviesAndTV/item.json
- datasets_orig/Amazon/MoviesAndTV/train.json
- datasets_orig/Amazon/MoviesAndTV/val.json
- datasets_orig/Amazon/MoviesAndTV/test.json

Writes:
- datasets_orig/Amazon/MoviesAndTV/user_items.json
"""

from __future__ import annotations

import json
from os import path
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = Path("datasets/Amazon/MoviesAndTV")
ITEM_PATH = BASE_DIR / "item.json"
SPLIT_FILES = [BASE_DIR / "train.jsonl", BASE_DIR / "validation.jsonl", BASE_DIR / "test.jsonl"]
OUTPUT_PATH = BASE_DIR / "user_items.jsonl"


def load_json_or_jsonl(path: Path) -> Any:
	text = path.read_text(encoding="utf-8").strip()
	if not text:
		return []
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		records = []
		for line in text.splitlines():
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
		return records


def iter_records(data: Any) -> Iterable[Dict[str, Any]]:
	if isinstance(data, list):
		for row in data:
			if isinstance(row, dict):
				yield row
	elif isinstance(data, dict):
		if any(isinstance(v, list) for v in data.values()):
			for value in data.values():
				if isinstance(value, list):
					for row in value:
						if isinstance(row, dict):
							yield row
		elif isinstance(data.get("data"), list):
			for row in data["data"]:
				if isinstance(row, dict):
					yield row
		elif isinstance(data.get("reviews"), list):
			for row in data["reviews"]:
				if isinstance(row, dict):
					yield row
		else:
			yield data


def extract_user_item_review_rating(record: Dict[str, Any]) -> Tuple[str, str, str, Any]:
	user = record.get("user") or record.get("user_id") or record.get("reviewerID") or record.get("reviewer_id")
	item = record.get("item") or record.get("item_id") or record.get("asin") or record.get("product_id") or record.get("parent_asin")
	review = (
		record.get("review")
		or record.get("reviewText")
		or record.get("text")
		or record.get("summary")
		or ""
	)
	rating = record.get("rating") or record.get("overall") or record.get("score")
	return str(user), str(item), str(review), rating


def load_item_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
	metadata: Dict[str, Dict[str, Any]] = {}
	for row in iter_records(load_json_or_jsonl(path)):
		item_id = row.get("item") or row.get("item_id") or row.get("asin") or row.get("product_id") or row.get("parent_asin")
		if item_id is None:
			continue
		description = row.get("description") or row.get("desc") or row.get("feature") or row.get("features") or row.get("brand") or ""
		if isinstance(description, list):
			description = " ".join(str(x) for x in description)
		metadata[str(item_id)] = {
			"title": row.get("title") or row.get("name") or row.get("summary") or "",
			"description": description,
		}
	return metadata


def build_user_items() -> Dict[str, List[Dict[str, Any]]]:
	item_metadata = load_item_metadata(ITEM_PATH) if ITEM_PATH.exists() else {}
	user_items: Dict[str, List[Dict[str, Any]]] = {}

	for split_path in SPLIT_FILES:
		if not split_path.exists():
			continue
		for record in iter_records(load_json_or_jsonl(split_path)):
			user, item, review, rating = extract_user_item_review_rating(record)
			if not user or not item or user == "None" or item == "None":
				continue
			if isinstance(review, list):
				review = " ".join(str(x) for x in review)
			elif review is None:
				review = ""
			meta = item_metadata.get(item, {})
			user_items.setdefault(user, []).append(
				{
					"item_id": item,
					"title": meta.get("title", ""),
					"description": meta.get("description", ""),
					"review": review,
					"rating": rating,
				}
			)

	return user_items


def main() -> None:
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	with OUTPUT_PATH.open("w", encoding="utf-8") as f:
		json.dump(build_user_items(), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
	main()
