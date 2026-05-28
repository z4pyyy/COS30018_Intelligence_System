"""
Generate a detailed PDF summary of the two-step improvement pipeline results.
"""
from fpdf import FPDF
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "docs"
OUT.mkdir(exist_ok=True)


class SummaryPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "COS30018 Fall Detection - Two-Step Pipeline Improvement Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        self.set_font("Helvetica", "", 9)
        self.set_text_color(0, 0, 0)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(240, 245, 250)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, str(val), border=1, fill=fill, align="C")
            self.ln()
        self.ln(3)


def main():
    pdf = SummaryPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Two-Step Pipeline Improvement Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "COS30018 Intelligent Systems  |  Fall Detection Project  |  12 May 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # 1. Executive Summary
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report documents the diagnosis and fix of a critical train-inference domain mismatch "
        "in the two-step fall detection pipeline. The original pipeline (v1) achieved only 46.1% "
        "overall accuracy with Walk recall collapsing to 20.4%. After applying consistent 25% "
        "bounding box padding across training and inference, the improved pipeline (v2) achieves "
        "69.6% overall accuracy with Walk recall recovering to 80.1%."
    )
    pdf.body_text(
        "The two-step pipeline (v2) now outperforms the one-step YOLOv8 baseline (67.7%) in "
        "overall accuracy and maintains a significant advantage in Sit detection (78.4% vs 16.5%). "
        "The one-step model retains superiority in Fall recall (92.9% vs 36.9%)."
    )

    # 2. Problem Diagnosis
    pdf.section_title("2. Problem Diagnosis: Train-Inference Domain Gap")
    pdf.sub_title("2.1 Root Cause")
    pdf.body_text(
        "The v1 pipeline extracted pose features from full training images but at inference time "
        "first cropped person regions using Stage 1 (YOLOv8n person detector) before feeding to "
        "MediaPipe. This meant MediaPipe saw completely different visual domains:\n\n"
        "- Training: Full scene images with background, multiple scale references\n"
        "- Inference: Tight person crops with no background context\n\n"
        "Walking poses were most affected because their tall, narrow bounding boxes lost limb "
        "context when cropped tightly. MediaPipe produced different joint positions relative to "
        "frame edges, different confidence levels, and different scale normalization."
    )

    pdf.sub_title("2.2 Evidence: Feature Distribution Shift")
    pdf.body_text("Key feature means showing the distribution shift between v1 (full images) and v2 (padded crops):")

    pdf.add_table(
        ["Feature", "v1 Fall", "v2 Fall", "v1 Walk", "v2 Walk", "Impact"],
        [
            ["spine_angle", "31.4", "41.2", "12.9", "16.5", "Wider separation"],
            ["body_aspect_ratio", "1.01", "1.16", "0.48", "0.73", "Clearer distinction"],
            ["hip_height_norm", "0.57", "0.50", "0.69", "0.61", "Falls appear lower"],
            ["knee_bend_left", "129.6", "119.8", "150.9", "145.9", "More discriminative"],
        ],
        col_widths=[38, 25, 25, 25, 25, 52]
    )

    # 3. Fix Applied
    pdf.section_title("3. Fix: Consistent Padded Crop Simulation")
    pdf.sub_title("3.1 Approach")
    pdf.body_text(
        "Applied BBOX_PADDING = 0.25 (25% expansion on each side) consistently across all four "
        "pipeline components:\n\n"
        "1. extract_pose_features.py: Read ground-truth YOLO bbox, pad by 25%, crop, then extract "
        "MediaPipe features from the padded crop (simulates inference domain)\n"
        "2. train_stage2_pose.py: Retrained MLP on v2 features from padded crops\n"
        "3. evaluate_twostep.py: Applied padding to Stage 1 detection boxes before cropping\n"
        "4. gui_fall_detection.py: Applied same padding to live webcam inference\n\n"
        "The pad_bbox() helper function clamps to frame boundaries and is identical in all files."
    )

    pdf.sub_title("3.2 Training Data Statistics")
    pdf.add_table(
        ["Metric", "v1 (Full Images)", "v2 (Padded Crops)"],
        [
            ["Training samples", "12,713", "12,114"],
            ["Test samples", "346", "346"],
            ["Failed (no pose)", "~3,000", "3,723"],
            ["Fall count", "4,766", "4,108"],
            ["Sit count", "3,584", "3,434"],
            ["Walk count", "4,363", "4,572"],
        ],
        col_widths=[55, 65, 65]
    )
    pdf.body_text(
        "Higher failure rate in v2 is expected: tighter crops from padded bounding boxes sometimes "
        "result in partial body views where MediaPipe cannot detect a full pose skeleton."
    )

    # 4. Results
    pdf.section_title("4. Results")
    pdf.sub_title("4.1 Standalone MLP Comparison (v1 vs v2)")
    pdf.add_table(
        ["Metric", "MLP v1 (Full Image)", "MLP v2 (Padded Crop)", "Delta"],
        [
            ["Overall Accuracy", "71.1%", "79.2%", "+8.1%"],
            ["Fall Recall", "-", "80.0%", "-"],
            ["Sit Recall", "-", "59.0%", "-"],
            ["Walk Recall", "~68%", "89.0%", "+21%"],
        ],
        col_widths=[42, 50, 50, 42]
    )

    pdf.sub_title("4.2 Full Pipeline Three-Way Comparison")
    pdf.body_text("This is the centrepiece comparison showing one-step vs two-step v1 vs two-step v2:")

    pdf.add_table(
        ["Metric", "One-Step", "Two-Step v1", "Two-Step v2", "v1->v2 Delta"],
        [
            ["Overall Accuracy", "67.7%", "46.1%", "69.6%", "+23.5%"],
            ["Fall Recall", "92.9%", "64.3%", "36.9%", "-27.4%"],
            ["Sit Recall", "16.5%", "78.4%", "78.4%", "0%"],
            ["Walk Recall", "83.4%", "20.4%", "80.1%", "+59.7%"],
            ["No-detect Rate", "0%", "14.1%", "11.6%", "-2.5%"],
        ],
        col_widths=[35, 32, 35, 35, 40]
    )

    pdf.sub_title("4.3 Per-Class Detailed Results (Two-Step v2 Pipeline)")
    pdf.add_table(
        ["Class", "Total", "Correct", "Accuracy", "Precision", "Recall", "F1"],
        [
            ["Fall", "84", "31", "36.9%", "0.7045", "0.3690", "0.4844"],
            ["Sit", "97", "76", "78.4%", "0.7379", "0.7835", "0.7600"],
            ["Walk", "181", "145", "80.1%", "0.8382", "0.8011", "0.8192"],
        ],
        col_widths=[25, 22, 25, 28, 30, 28, 25]
    )

    pdf.sub_title("4.4 Stage Breakdown")
    pdf.add_table(
        ["Stage", "Count", "Percentage"],
        [
            ["Pose (full pipeline)", "320", "88.4%"],
            ["No person detected", "32", "8.8%"],
            ["No pose detected", "10", "2.8%"],
        ],
        col_widths=[60, 60, 60]
    )

    # 5. Analysis
    pdf.add_page()
    pdf.section_title("5. Analysis and Discussion")

    pdf.sub_title("5.1 Walk Recall Recovery")
    pdf.body_text(
        "Walk recall improved from 20.4% (v1) to 80.1% (v2), a 59.7 percentage point gain. "
        "This validates the hypothesis that the domain gap was the primary cause of Walk classification "
        "failure. Walking poses in tall, narrow bounding boxes now produce consistent features between "
        "training and inference because both apply the same 25% padding."
    )

    pdf.sub_title("5.2 Fall Recall Regression")
    pdf.body_text(
        "Fall recall dropped from 64.3% (v1) to 36.9% (v2). This is a tradeoff from improved "
        "Walk/Sit discrimination. Possible causes:\n\n"
        "- Falls often have person partially on the ground with atypical body geometry\n"
        "- Stage 1 detection of fallen persons may produce less accurate bounding boxes\n"
        "- Padded crops of fallen persons still differ from training ground-truth padded crops\n"
        "- The MLP now better distinguishes Walk from Fall, but some ambiguous Falls are classified "
        "as Sit (both involve horizontal/low body positions)\n\n"
        "Future work: Add fall-specific data augmentation, use temporal features from video sequences, "
        "or add a dedicated fall confidence threshold."
    )

    pdf.sub_title("5.3 One-Step vs Two-Step Trade-offs")
    pdf.body_text(
        "One-step (YOLOv8): Best for Fall detection (92.9% recall), simple architecture, "
        "zero-latency detection. Weakness: poor Sit discrimination (16.5%).\n\n"
        "Two-step v2 (YOLO + Pose + MLP): Best for Sit (78.4%) and Walk (80.1%), more balanced "
        "overall accuracy (69.6%). Weakness: Fall recall (36.9%), higher latency, 11.6% no-detection "
        "rate from Stage 1 misses.\n\n"
        "A production system could combine both: use one-step for Fall alerts (high recall critical) "
        "and two-step for activity classification (Sit/Walk distinction matters for elder care "
        "monitoring)."
    )

    pdf.sub_title("5.4 Remaining Limitations")
    pdf.body_text(
        "1. No-detection rate (11.6%): Stage 1 person detector misses some persons, especially in "
        "unusual poses or low-light conditions.\n"
        "2. Fall recall: The two-step pipeline struggles with Falls, likely due to Stage 1 bbox "
        "quality for fallen persons.\n"
        "3. Test set size: 362 images is relatively small for robust statistical conclusions.\n"
        "4. No temporal features: Single-frame classification cannot capture fall dynamics "
        "(velocity, acceleration).\n"
        "5. MediaPipe pose detection failures: 3,723 training images (23.5%) failed pose extraction, "
        "reducing training data diversity."
    )

    # 6. Architecture
    pdf.section_title("6. System Architecture")
    pdf.body_text(
        "Two-Step Pipeline Flow:\n\n"
        "Input Frame -> Stage 1 (YOLOv8n Person Detector, conf=0.35)\n"
        "  -> Person Bounding Box\n"
        "  -> pad_bbox(25% expansion on each side)\n"
        "  -> Crop padded region from frame\n"
        "  -> MediaPipe Pose Landmarker (Heavy model)\n"
        "  -> Extract 8 geometric features\n"
        "  -> StandardScaler normalization\n"
        "  -> PoseMLP (8->128->64->3)\n"
        "  -> Softmax -> Fall/Sit/Walk classification\n\n"
        "One-Step Pipeline Flow:\n\n"
        "Input Frame -> YOLOv8 (swook_techcare model, conf=0.45)\n"
        "  -> Direct Fall/Sit/Walk bounding boxes + confidence"
    )

    pdf.sub_title("6.1 Model Details")
    pdf.add_table(
        ["Component", "Architecture", "Parameters", "File"],
        [
            ["Stage 1", "YOLOv8n", "3.2M", "stage1_person.pt"],
            ["Pose Model", "MediaPipe Heavy", "~30M", "pose_landmarker_heavy.task"],
            ["Stage 2 MLP", "Linear(8,128)->BN->ReLU->DO(0.3)->Linear(128,64)->BN->ReLU->DO(0.2)->Linear(64,3)", "~10K", "stage2_pose_mlp_v2.pt"],
            ["One-Step", "YOLOv8 (swook)", "~6M", "swook_techcare_fall_sit_walk.pt"],
        ],
        col_widths=[28, 85, 25, 50]
    )

    # 7. Files Modified
    pdf.section_title("7. Files Modified in This Improvement")
    pdf.add_table(
        ["File", "Change", "Status"],
        [
            ["src/extract_pose_features.py", "Added padded crop simulation via pad_bbox()", "Modified"],
            ["src/train_stage2_pose.py", "Updated to v2 CSV/model paths", "Modified"],
            ["src/evaluate_twostep.py", "Added pad_bbox() + v2 model paths", "Modified"],
            ["src/gui_fall_detection.py", "Added pad_bbox() + v2 model paths", "Modified"],
            ["datasets/pose_features/train_features_v2.csv", "12,114 padded crop features", "New"],
            ["datasets/pose_features/test_features_v2.csv", "346 test features", "New"],
            ["models/stage2_pose_mlp_v2.pt", "Retrained MLP weights", "New"],
            ["models/pose_scaler_v2.pkl", "Feature scaler for v2", "New"],
        ],
        col_widths=[70, 75, 30]
    )

    # Save
    out_path = OUT / "twostep_improvement_summary.pdf"
    pdf.output(str(out_path))
    print(f"PDF saved -> {out_path}")


if __name__ == "__main__":
    main()
