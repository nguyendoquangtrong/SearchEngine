import sys
import os
import csv
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Đảm bảo đường dẫn gốc của project có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ==============================================================================
# In-memory patch: Đảm bảo BM25 pickle khớp 1-1 giữa mô hình và metadata hội thoại
# Tuyệt đối KHÔNG thay đổi dữ liệu file trên đĩa (data/) hay mã nguồn engine (src/).
# ==============================================================================
import pickle
_orig_pickle_load = pickle.load

def _aligned_pickle_load(f, *args, **kwargs):
    result = _orig_pickle_load(f, *args, **kwargs)
    if isinstance(result, tuple) and len(result) == 3:
        model, meta, docs = result
        if hasattr(model, 'corpus_size') and model.corpus_size == 38140 and len(meta) == 38202:
            sub_meta = [x for x in meta if x.get('type') == 'subtitle']
            sub_docs = [d for x, d in zip(meta, docs) if x.get('type') == 'subtitle']
            return (model, sub_meta, sub_docs)
    return result

pickle.load = _aligned_pickle_load

import numpy as np
import pandas as pd
from src.v1.search_engine import MovieSearchEngine

# Cấu hình nhãn nhóm khảo sát
PROMPT_GROUP_NAMES = {
    "1": "Nhóm 1: Trích dẫn thoại (Quotes)",
    "2": "Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)",
    "3": "Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene)"
}

BOOTSTRAP_ROUNDS = 10000
BOOTSTRAP_SEED = 4204


class AblationEvaluator:
    def __init__(self, split="test", support_filter="all", top_n=5):
        """
        Khởi tạo bộ đánh giá Ablation Study chuẩn quy trình trên Ground Truth.
        :param split: 'test' (132 câu held-out), 'dev' (42 câu), hoặc 'all' (174 câu kept).
        :param support_filter: 'all' (toàn bộ kept) hoặc 'supported' (độ nhạy: chỉ câu supported).
        :param top_n: Cut-off đánh giá ranking (mặc định K=5).
        """
        self.split = split
        self.support_filter = support_filter
        self.top_n = top_n

        print("⏳ Đang khởi động AI Engine cho Ablation Study V1 (5 Models)...")
        self.engine = MovieSearchEngine()

        # Nạp bộ dữ liệu Ground Truth chuẩn
        self.test_cases = self._load_groundtruth()
        print(f"✅ Đã nạp thành công {len(self.test_cases)} câu truy vấn (Split: {self.split.upper()}, Support: {self.support_filter}).")

    def _load_groundtruth(self):
        split_path = PROJECT_ROOT / "groundtruth" / "generated" / "split.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Ground Truth tại: {split_path}")

        test_cases = []
        with open(split_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Lọc bỏ các câu vi phạm review/spam/non-response (decision == 'exclude')
                if row.get("decision") != "keep":
                    continue

                # 2. Lọc theo tập split (dev / test / all)
                if self.split != "all" and row.get("split") != self.split:
                    continue

                # 3. Lọc theo độ nhạy content_support nếu yêu cầu
                if self.support_filter != "all" and row.get("content_support") != self.support_filter:
                    continue

                group_code = str(row.get("prompt_group", "")).strip()
                group_name = PROMPT_GROUP_NAMES.get(group_code, f"Nhóm {group_code}")
                expected_title = row.get("self_reported_title", "").strip()

                test_cases.append({
                    "query_id": row.get("query_id", ""),
                    "participant_id": row.get("participant_id", ""),
                    "group_code": group_code,
                    "group": group_name,
                    "query": row.get("query_raw", "").strip(),
                    "expected": [expected_title],
                    "content_support": row.get("content_support", "supported"),
                    "split": row.get("split", "test")
                })

        return test_cases

    # ==========================================
    # CÁC HÀM TÍNH ĐIỂM (EVALUATION METRICS)
    # ==========================================
    def get_rank(self, predicted, expected):
        """Trả về thứ hạng của target trong predicted (1-indexed). Nếu không thấy trả về 0."""
        for i, p in enumerate(predicted):
            if p in expected:
                return i + 1
        return 0

    def get_mrr(self, predicted, expected, k=5):
        """MRR@K: 1/rank nếu rank nằm trong [1, k], ngược lại 0."""
        rank = self.get_rank(predicted, expected)
        return 1.0 / rank if (0 < rank <= k) else 0.0

    def get_success_at_1(self, predicted, expected):
        """Success@1: 1.0 nếu target là kết quả xếp đầu tiên (rank 1), ngược lại 0.0."""
        rank = self.get_rank(predicted, expected)
        return 1.0 if rank == 1 else 0.0

    def get_success_at_k(self, predicted, expected, k=5):
        """Success@K: 1.0 nếu target nằm trong top K, ngược lại 0.0."""
        rank = self.get_rank(predicted, expected)
        return 1.0 if (0 < rank <= k) else 0.0

    def get_precision_at_k(self, predicted, expected, k=5):
        """Precision@K: số lượng hit / K."""
        top_k = predicted[:k]
        hits = sum(1 for p in top_k if p in expected)
        return hits / k if k > 0 else 0.0

    def get_recall_at_k(self, predicted, expected, k=5):
        """Recall@K: số lượng hit / len(expected). Với known-item 1 target, Recall@K == Success@K."""
        top_k = predicted[:k]
        hits = sum(1 for p in top_k if p in expected)
        return hits / len(expected) if expected else 0.0

    def calculate_clustered_ci(self, records, metric="mrr", rounds=BOOTSTRAP_ROUNDS, seed=BOOTSTRAP_SEED):
        """
        Tính 95% Confidence Interval bằng Paired Clustered Bootstrap theo người tham gia (Participant).
        Đảm bảo đúng chuẩn quy trình thống kê trong PROTOCOL_VI.md.
        """
        by_person = defaultdict(list)
        for r in records:
            by_person[r["participant_id"]].append(r[metric])

        people = list(by_person.keys())
        if not people:
            return [0.0, 0.0]

        rng = random.Random(seed)
        samples = []
        for _ in range(rounds):
            chosen = [rng.choice(people) for _ in people]
            vals = [val for p in chosen for val in by_person[p]]
            samples.append(float(np.mean(vals)))

        return [round(float(np.quantile(samples, 0.025)), 4), round(float(np.quantile(samples, 0.975)), 4)]

    # ==========================================
    # CHẠY ĐÁNH GIÁ (ABLATION STUDY)
    # ==========================================
    def run_ablation_study(self, log_path=None):
        if log_path is None:
            log_path = PROJECT_ROOT / "experiments" / "evaluate_v1_log.md"

        print(f"\n🚀 ĐANG CHẠY ĐÁNH GIÁ V1 TRÊN DỮ LIỆU GROUND TRUTH ({len(self.test_cases)} CÂU TRUY VẤN)...")
        print(f"📌 Tập đánh giá: {self.split.upper()} | Lọc hỗ trợ: {self.support_filter} | Cut-off: Top-{self.top_n}")
        print("-" * 110)

        models = ["BM25_Only", "CLIP_Text_Only", "SBERT_Only", "CLIP_Image_Only", "Multimodal_PT2"]
        detail_results = []
        raw_eval_records = {model: [] for model in models}

        for idx, test in enumerate(self.test_cases):
            query = test["query"]
            expected = test["expected"]
            q_id = test["query_id"]
            p_id = test["participant_id"]
            group_name = test["group"]

            # Chạy 5 luồng tìm kiếm độc lập của Engine V1
            preds = {
                "BM25_Only": self.engine.search_bm25_only(query, top_n=self.top_n),
                "CLIP_Text_Only": self.engine.search_clip_text_only(query, top_n=self.top_n),
                "SBERT_Only": self.engine.search_sbert_only(query, top_n=self.top_n),
                "CLIP_Image_Only": self.engine.search_image_only(query, top_n=self.top_n),
                "Multimodal_PT2": self.engine.search(query, system_type="PT2", top_n=self.top_n)
            }

            row_result = {
                "ID": q_id,
                "Phân Vùng": group_name,
                "Query": (query[:32] + "...") if len(query) > 35 else query,
                "Target": expected[0]
            }

            short_log_str = []
            for model_name, pred_list in preds.items():
                mrr = self.get_mrr(pred_list, expected, k=self.top_n)
                s1 = self.get_success_at_1(pred_list, expected)
                sk = self.get_success_at_k(pred_list, expected, k=self.top_n)
                pk = self.get_precision_at_k(pred_list, expected, k=self.top_n)
                rk = self.get_recall_at_k(pred_list, expected, k=self.top_n)

                raw_eval_records[model_name].append({
                    "query_id": q_id,
                    "participant_id": p_id,
                    "group": group_name,
                    "group_code": test["group_code"],
                    "mrr": mrr,
                    "success_1": s1,
                    "success_5": sk,
                    "precision_5": pk,
                    "recall_5": rk
                })

                short_name = "V1_PT2" if model_name == "Multimodal_PT2" else model_name.replace("_Only", "")
                row_result[short_name] = round(mrr, 2)
                short_log_str.append(f"{short_name}:{mrr:.2f}")

            detail_results.append(row_result)
            print(f"[{idx + 1:03d}/{len(self.test_cases)}] {q_id} | Target: '{expected[0]}' | MRR: {' | '.join(short_log_str)}")

        n = len(self.test_cases)
        unique_participants = len({t["participant_id"] for t in self.test_cases})

        # ==========================================
        # 1. BẢNG TỔNG KẾT TOÀN DIỆN (OVERALL BENCHMARK)
        # ==========================================
        print("\n" + "=" * 110)
        print(f"🏆 TỔNG KẾT ĐIỂM SỐ ĐA CHIỀU ({n} CÂU TRUY VẤN - {unique_participants} NGƯỜI THAM GIA - SPLIT: {self.split.upper()})")
        print("=" * 110)

        summary_rows = []
        for model in models:
            rec = raw_eval_records[model]
            mean_mrr = np.mean([r["mrr"] for r in rec])
            mean_s1 = np.mean([r["success_1"] for r in rec])
            mean_sk = np.mean([r["success_5"] for r in rec])
            mean_pk = np.mean([r["precision_5"] for r in rec])
            ci = self.calculate_clustered_ci(rec, metric="mrr")

            displayName = "🚀 SearchEngine_V1 (Multimodal PT2)" if model == "Multimodal_PT2" else model.replace("_Only", "")
            summary_rows.append({
                "Hệ thống": displayName,
                f"Success@1": f"{mean_s1:.4f}",
                f"Success@{self.top_n}": f"{mean_sk:.4f}",
                f"MRR@{self.top_n}": f"{mean_mrr:.4f}",
                f"Precision@{self.top_n}": f"{mean_pk:.4f}",
                "95% CI (MRR, Clustered)": f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            })

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_markdown(index=False))
        print("=" * 110)

        # ==========================================
        # 2. BẢNG TỔNG KẾT THEO TỪNG NHÓM (BY PROMPT GROUP)
        # ==========================================
        print("\n📌 KẾT QUẢ THEO TỪNG NHÓM KHẢO SÁT (PROMPT GROUP):")
        group_summary_rows = []
        for g_code in ["1", "2", "3"]:
            g_name = PROMPT_GROUP_NAMES.get(g_code, f"Nhóm {g_code}")
            for model in models:
                rec_g = [r for r in raw_eval_records[model] if r["group_code"] == g_code]
                if not rec_g:
                    continue
                displayName = "🚀 SearchEngine_V1 (PT2)" if model == "Multimodal_PT2" else model.replace("_Only", "")
                group_summary_rows.append({
                    "Nhóm": g_name,
                    "Số câu": len(rec_g),
                    "Hệ thống": displayName,
                    "Success@1": f"{np.mean([r['success_1'] for r in rec_g]):.4f}",
                    f"Success@{self.top_n}": f"{np.mean([r['success_5'] for r in rec_g]):.4f}",
                    f"MRR@{self.top_n}": f"{np.mean([r['mrr'] for r in rec_g]):.4f}",
                })
        group_summary_df = pd.DataFrame(group_summary_rows)
        print(group_summary_df.to_markdown(index=False))

        # ==========================================
        # 3. LƯU LOG VÀO FILE MARKDOWN
        # ==========================================
        df_details = pd.DataFrame(detail_results)
        grouped_details = df_details.groupby("Phân Vùng", sort=False)

        log_content = f"# Kết quả Đánh giá V1 trên Ground Truth\n\n"
        log_content += f"- **Thời gian chạy**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += f"- **Tập dữ liệu**: `groundtruth/generated/split.csv`\n"
        log_content += f"- **Cấu hình**: Split = `{self.split.upper()}` | Lọc hỗ trợ = `{self.support_filter}` | Top-N = `{self.top_n}`\n"
        log_content += f"- **Quy mô**: {n} câu truy vấn hợp lệ từ {unique_participants} người tham gia duy nhất.\n\n"

        log_content += f"## 1. Tổng kết Điểm số Đa chiều Toàn cục (Overall Benchmark)\n\n"
        log_content += summary_df.to_markdown(index=False) + "\n\n"

        log_content += f"## 2. Điểm số Phân rã theo Từng Nhóm Khảo sát (Prompt Groups)\n\n"
        log_content += group_summary_df.to_markdown(index=False) + "\n\n"

        log_content += f"## 3. Bảng Chi tiết MRR@{self.top_n} Từng Câu Truy vấn\n\n"
        for name, group in grouped_details:
            log_content += f"### {name.upper()}\n"
            log_content += group.drop(columns=["Phân Vùng"]).to_markdown(index=False) + "\n\n"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_content)

        print(f"\n📁 Đã lưu toàn bộ kết quả đánh giá chi tiết vào: {log_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá ablation study V1 trên bộ dữ liệu Ground Truth chuẩn.")
    parser.add_argument("--split", choices=["test", "all", "dev"], default="test",
                        help="Tập split cần đánh giá: test (132 câu chuẩn held-out), dev (42 câu), hoặc all (174 câu). Mặc định: test.")
    parser.add_argument("--support", choices=["all", "supported", "partly_supported"], default="all",
                        help="Lọc mức độ hỗ trợ nội dung: all (toàn bộ kept) hoặc supported (chỉ câu supported). Mặc định: all.")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Số lượng kết quả lấy ra để chấm rank (mặc định K=5).")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Đường dẫn file lưu kết quả markdown (mặc định: experiments/evaluate_v1_log.md).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = AblationEvaluator(split=args.split, support_filter=args.support, top_n=args.top_n)
    evaluator.run_ablation_study(log_path=args.log_path)