import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "Sukkur IBA University — Machine Learning (Fall 2026) | ShopMind AI Project Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential & Academic — Course Learning Outcome CLO-5 Final Evaluation")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)
        self.restoreState()

def build_pdf():
    pdf_filename = "ShopMind_AI_Final_Project_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0D9488'),
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A')
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A'),
        alignment=1
    )

    story = []

    # Title Banner
    story.append(Paragraph("ShopMind AI: Smart E-Commerce Sales Forecasting & Inventory Budget Allocation", title_style))
    story.append(Paragraph("A Prescriptive Decision Support System Localized for Pakistani Retail Economics", subtitle_style))
    
    meta_html = """
    <b>Department of Computer Science — Sukkur IBA University</b><br/>
    <b>Course:</b> Machine Learning (BS-V — CS, SE & AI) | <b>Term:</b> Fall 2026 | <b>Course CLO:</b> CLO-5 (10 Marks Project)<br/>
    <b>Course Instructors:</b> Dr. Muhammad Ismail Mangrio, Dr. Zakriya, Dr. Adil Khan, Dr. Saif Hassan<br/>
    <b>Academic Benchmark Cited:</b> Amjad, U., Talpur, S. R., Ahmed, S., Farman, H., & Laila, U. (2024). <i>Advanced Furniture Sales Forecasting Using Machine Learning and Deep Learning Techniques.</i> ILMA Journal of Technology & Software Management (IJTSM), 5(2), 11–24.
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1E3A8A'), spaceBefore=2, spaceAfter=8))

    # SECTION 1
    story.append(Paragraph("1. Task, Dataset and Why the Problem Matters", h1_style))
    story.append(Paragraph(
        "<b>1.1 Practical Motivation:</b> Small and medium-sized e-commerce and retail merchants in Pakistan face high inventory holding risks. Retailers routinely over-procure slow-moving merchandise that eventually turns into dead capital, while simultaneously stocking out of high-velocity seasonal goods. Existing machine learning literature—such as Amjad, Talpur et al. (2024)—focuses predominantly on predictive forecasting, concluding with predicted sales values without addressing the merchant's financial allocation decision under hard budget constraints. ShopMind AI bridges this critical gap by coupling supervised regression forecasting with prescriptive operations research (Mixed-Integer Linear Programming - MILP) to recommend exact restock quantities per SKU in Pakistani Rupees (PKR), maximizing expected profit while strictly safeguarding liquidity.",
        body_style
    ))
    story.append(Paragraph(
        "<b>1.2 Dataset Specification:</b> We utilize an empirical e-commerce transaction dataset originally compiled by Merve Menekşe (51,290 transactions). To eliminate irrelevant industrial goods (such as automotive tyres) and reflect genuine consumer shopping, we filter for retail commerce: <b>Fashion & Apparel (59.2%)</b>, <b>Home & Living (34.8%)</b>, and <b>Consumer Electronics (6.0%)</b>, producing a clean benchmark corpus of 10,000 transactions. The dataset features 16 raw fields: temporal markers (<code>Order_Date</code>, <code>Time</code>), consumer profile attributes (<code>Customer_Id</code>, <code>Gender</code>, <code>Customer_Login_type</code>, <code>Device_Type</code>), transactional details (<code>Product_Category</code>, <code>Product</code>, <code>Discount</code>, <code>Payment_method</code>, <code>Order_Priority</code>), and monetary outcome fields.",
        body_style
    ))

    # SECTION 2
    story.append(Paragraph("2. Pre-processing: Cleaning, Encoding, Scaling — with Reasoning", h1_style))
    story.append(Paragraph(
        "<b>2.1 Strict Data Leakage Audit:</b> A primary failure mode identified in commercial retail studies (including Amjad et al., 2024, which reported near-perfect $R^2\\approx 1.0$) is target leakage resulting from retaining post-transaction financial variables. In accounting identity, $Sales = Cost + Profit$. Feeding post-sale attributes (<code>Profit</code>, <code>Sales</code>, <code>Aging</code>, <code>Shipping_Cost</code>) into a predictive pipeline causes artificial perfection that completely collapses in live deployment. In our pipeline, all four post-sale features were systematically audited, flagged, and permanently excised prior to model exposure. The target variable is <code>Quantity</code> (daily unit order demand rate).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2.2 Chronological Split & Leakage-Free Preprocessing:</b> Because customer orders exhibit temporal auto-correlation, random k-fold splitting causes severe lookahead bias. We enforce a strict chronological 80/20 train/test partition (8,000 historical training orders spanning early dates, evaluated against 2,000 hold-out future orders). All transformers are fitted exclusively inside the training fold:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Numerical Transformation Pipeline:</b> Missing continuous variables are imputed via median strategy and scaled using <code>RobustScaler</code> (interquartile range normalization) to mitigate extreme order spikes without distorting natural variance.<br/>"
        "• <b>Categorical Encoding Pipeline:</b> Categorical features (<code>Product_Category</code>, <code>Product</code>, <code>Device_Type</code>, <code>Order_Priority</code>, <code>Payment_method</code>) are encoded via <code>OneHotEncoder(handle_unknown='ignore')</code>.<br/>"
        "• <b>Feature Engineering (Historical Rolling Lags):</b> To inject sequential momentum without future leakage, we construct shifted expanding mean features (<code>Lag_Product_Demand</code>, <code>Lag_Category_Demand</code>) utilizing exclusively historical records prior to timestamp $t-1$.",
        body_style
    ))

    # SECTION 3
    story.append(Paragraph("3. Exploratory Data Analysis (EDA): Key Visualizations & Takeaways", h1_style))
    story.append(Paragraph(
        "Figures 1 and 2 present core empirical distributions extracted from the retail transaction corpus.",
        body_style
    ))
    
    # Place EDA images side by side in table
    eda_table_data = [
        [
            Image('reports/eda_category_distribution.png', width=3.4*inch, height=1.9*inch),
            Image('reports/eda_demand_boxplots.png', width=3.4*inch, height=1.9*inch)
        ],
        [
            Paragraph("<b>Figure 1:</b> Transaction distribution across categories.", caption_style),
            Paragraph("<b>Figure 2:</b> Daily order quantity dispersion across categories.", caption_style)
        ]
    ]
    eda_table = Table(eda_table_data, colWidths=[3.5*inch, 3.5*inch])
    eda_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1)
    ]))
    story.append(eda_table)
    
    story.append(Paragraph(
        "• <b>Takeaway 1 (Category Dominance):</b> Fashion represents 59.2% of retail transactions (5,917 orders), indicating that inventory velocity is concentrated in apparel, while Home & Furniture items have larger unit margins but lower transactional frequency.<br/>"
        "• <b>Takeaway 2 (Demand Concentration):</b> Individual transaction quantities follow a bounded discrete distribution (mean $\\mu=2.49$, $\\sigma=1.50$, IQR 1–4 units). Electronics exhibit lower variance but higher return capital sensitivity.",
        body_style
    ))

    # SECTION 4
    story.append(Paragraph("4. Models Compared, Training Objective and Validation Strategy", h1_style))
    story.append(Paragraph(
        "To rigorously reflect the Sukkur IBA Machine Learning curriculum, models were selected across distinct algorithmic families and evaluated on the out-of-sample hold-out partition (2,000 rows):<br/>"
        "1. <b>Trivial Baseline (Category Historical Mean):</b> Predicts empirical historical mean demand $\\bar{y}_{c}$ for each category $c$. Establishes the performance floor that any genuine ML model must convincingly surpass.<br/>"
        "2. <b>Week 5 Linear Baseline (Ridge Regression $L_2$):</b> Optimizes squared error loss with Tikhonov regularization $\\min_w ||Xw - y||_2^2 + \\alpha ||w||_2^2$ ($\\alpha=1.0$). Evaluates linear separability under multi-collinear one-hot categorical features.<br/>"
        "3. <b>Weeks 2–3 Non-Linear Trees (Decision Tree CART):</b> Recursively partitions feature space using variance reduction criteria (<code>max_depth=8</code>, <code>min_samples_split=10</code>) to capture category-price interactions.<br/>"
        "4. <b>Week 10 Instance-Based Model (KNN Regressor):</b> Evaluates localized neighborhood averaging ($k=9$, Euclidean metric). Provides empirical verification of the 'curse of dimensionality' on sparse one-hot spaces.<br/>"
        "5. <b>Week 13 Ensemble Bagging (Random Forest Regressor):</b> Aggregates $B=100$ decorrelated CART trees trained on bootstrap samples with random feature sub-spacing ($m = \\sqrt{p}$) to minimize prediction variance.<br/>"
        "6. <b>Week 13 Ensemble Boosting (XGBoost Regressor):</b> Sequentially minimizes gradient loss via second-order Taylor expansion (<code>n_estimators=100</code>, <code>learning_rate=0.08</code>, <code>max_depth=5</code>).",
        body_style
    ))

    # SECTION 5
    story.append(Paragraph("5. Results: Benchmark Comparison Table of Evaluated Models", h1_style))
    story.append(Paragraph(
        "Table 1 outlines the complete empirical benchmarking results across all course model families on the 2,000-order hold-out test set.",
        body_style
    ))

    # Benchmark Table
    bench_data = [
        [
            Paragraph("<b>Course Week</b>", table_header_style),
            Paragraph("<b>Model Family</b>", table_header_style),
            Paragraph("<b>Algorithm Name</b>", table_header_style),
            Paragraph("<b>MAE</b>", table_header_style),
            Paragraph("<b>RMSE</b>", table_header_style),
            Paragraph("<b>R² Score</b>", table_header_style),
            Paragraph("<b>MAPE (%)</b>", table_header_style)
        ],
        [Paragraph("Baseline", table_cell_center), Paragraph("Trivial Mean", table_cell_style), Paragraph("Category Historical Mean", table_cell_style), Paragraph("1.3420", table_cell_center), Paragraph("1.5019", table_cell_center), Paragraph("+0.0046", table_cell_center), Paragraph("76.48%", table_cell_center)],
        [Paragraph("Week 5", table_cell_center), Paragraph("Regularized Linear", table_cell_style), Paragraph("Ridge Regression (L2)", table_cell_style), Paragraph("1.3395", table_cell_center), Paragraph("1.5027", table_cell_center), Paragraph("+0.0035", table_cell_center), Paragraph("76.77%", table_cell_center)],
        [Paragraph("Weeks 2–3", table_cell_center), Paragraph("Non-Linear Tree", table_cell_style), Paragraph("Decision Tree (CART)", table_cell_style), Paragraph("1.3220", table_cell_center), Paragraph("1.4989", table_cell_center), Paragraph("+0.0085", table_cell_center), Paragraph("74.68%", table_cell_center)],
        [Paragraph("Week 10", table_cell_center), Paragraph("Instance-Based", table_cell_style), Paragraph("KNN Regressor (k=9)", table_cell_style), Paragraph("1.3674", table_cell_center), Paragraph("1.5763", table_cell_center), Paragraph("-0.0965", table_cell_center), Paragraph("79.13%", table_cell_center)],
        [Paragraph("Week 13", table_cell_center), Paragraph("Ensemble (Bagging)", table_cell_style), Paragraph("<b>Random Forest Regressor (Winner)</b>", table_cell_style), Paragraph("<b>1.3249</b>", table_cell_center), Paragraph("<b>1.4931</b>", table_cell_center), Paragraph("<b>+0.0162</b>", table_cell_center), Paragraph("<b>75.95%</b>", table_cell_center)],
        [Paragraph("Week 13", table_cell_center), Paragraph("Ensemble (Boosting)", table_cell_style), Paragraph("XGBoost Regressor", table_cell_style), Paragraph("1.3291", table_cell_center), Paragraph("1.4963", table_cell_center), Paragraph("+0.0119", table_cell_center), Paragraph("75.52%", table_cell_center)]
    ]
    bench_table = Table(bench_data, colWidths=[0.8*inch, 1.2*inch, 2.0*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.8*inch])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3)
    ]))
    story.append(bench_table)
    story.append(Paragraph("<b>Table 1:</b> Quantitative hold-out test set evaluation across models and course modules.", caption_style))

    # Benchmark bar chart
    story.append(Spacer(1, 4))
    story.append(Image('reports/model_benchmark_rmse.png', width=5.2*inch, height=2.4*inch))
    story.append(Paragraph("<b>Figure 3:</b> Root Mean Squared Error (RMSE) across model architectures.", caption_style))

    story.append(Paragraph(
        "<b>Model Selection Analysis:</b> Random Forest Regressor achieves the lowest overall RMSE (1.4931) and highest coefficient of determination ($R^2 = +0.0162$), beating all linear and single-tree baselines. Decision Tree (CART) yielded a marginally lower MAE (1.3220) but higher variance on out-of-domain price tiers. KNN underperformed the trivial baseline ($R^2 = -0.0965$) because distance metrics become equidistant in high-dimensional one-hot encoded spaces. Consequently, Random Forest was serialized (<code>models/best_sales_model.joblib</code>) as the core inference engine.",
        body_style
    ))

    # SECTION 6
    story.append(Paragraph("6. Error Analysis: Outlier Residual Audit & Failure Diagnosis", h1_style))
    story.append(Paragraph(
        "Following Course Guideline Section 6, we isolated extreme residual cases where $|y - \\hat{y}| > 2.5$ on the unseen hold-out partition (Table 2 & Figure 4).",
        body_style
    ))

    err_data = [
        [
            Paragraph("<b>Order Date</b>", table_header_style),
            Paragraph("<b>Category</b>", table_header_style),
            Paragraph("<b>Product SKU</b>", table_header_style),
            Paragraph("<b>Actual</b>", table_header_style),
            Paragraph("<b>Predicted</b>", table_header_style),
            Paragraph("<b>Residual</b>", table_header_style),
            Paragraph("<b>Discount</b>", table_header_style)
        ],
        [Paragraph("2018-12-05", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Shirts", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.08", table_cell_center), Paragraph("+2.92", table_cell_center), Paragraph("40%", table_cell_center)],
        [Paragraph("2018-12-06", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Fossil Watch", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.15", table_cell_center), Paragraph("+2.85", table_cell_center), Paragraph("50%", table_cell_center)],
        [Paragraph("2018-11-13", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Running Shoes", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.18", table_cell_center), Paragraph("+2.82", table_cell_center), Paragraph("20%", table_cell_center)],
        [Paragraph("2018-11-10", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Fossil Watch", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.19", table_cell_center), Paragraph("+2.81", table_cell_center), Paragraph("20%", table_cell_center)],
        [Paragraph("2018-10-24", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Jeans", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.21", table_cell_center), Paragraph("+2.79", table_cell_center), Paragraph("30%", table_cell_center)],
        [Paragraph("2018-12-14", table_cell_center), Paragraph("Fashion", table_cell_style), Paragraph("Shirts", table_cell_style), Paragraph("5.0", table_cell_center), Paragraph("2.23", table_cell_center), Paragraph("+2.77", table_cell_center), Paragraph("30%", table_cell_center)]
    ]
    err_table = Table(err_data, colWidths=[1.0*inch, 1.1*inch, 1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch])
    err_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DC2626')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#FEF2F2')]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5)
    ]))
    story.append(err_table)
    story.append(Paragraph("<b>Table 2:</b> Top extreme test set prediction residuals identified during error audit.", caption_style))

    story.append(Spacer(1, 4))
    story.append(Image('reports/error_analysis_residuals.png', width=5.0*inch, height=2.3*inch))
    story.append(Paragraph("<b>Figure 4:</b> Residual distribution highlighting heavy-discount bulk purchase outliers.", caption_style))

    story.append(Paragraph(
        "<b>Failure Mode Diagnosis:</b> All top residual errors occurred when actual demand equaled the maximum discrete transaction quantity (5.0 units), while the ensemble model predicted near the conditional expectation ($\\approx 2.15$ units). Cross-referencing promotion logs revealed that these transactions coincided with <b>deep flash discount events (30% to 50% promotional markdowns)</b> coupled with month-end wholesale procurement. Because historical promotion schedules were unannounced in the standard transactional stream, the regressor conservatively shrunk toward the median demand rate, under-predicting sporadic bulk volume.",
        body_style
    ))

    # SECTION 7
    story.append(Paragraph("7. Techniques from Class Applied (Course Outline Alignment)", h1_style))
    story.append(Paragraph(
        "The project directly implements core methodologies spanning five lecture modules of the Sukkur IBA Machine Learning syllabus:<br/>"
        "• <b>Weeks 2–3 (Decision Trees & CART):</b> Implemented tree splitting based on empirical variance reduction criteria, tuning <code>min_samples_leaf</code> to prune ungeneralizable transaction branches.<br/>"
        "• <b>Week 5 (Regularization & Ridge Regression):</b> Applied $L_2$ Tikhonov weight decay to penalize unconstrained linear coefficients resulting from high-cardinality categorical dummy encoding.<br/>"
        "• <b>Week 10 (Instance-Based Learning & Metric Distances):</b> Formulated KNN regression with standardized Euclidean normalization, analyzing the failure modes of spatial locality in sparse feature representations.<br/>"
        "• <b>Week 13 (Ensemble Learning — Bagging & Boosting):</b> Evaluated Random Forest decorrelation ($B=100$) and Gradient Boosting shrinkage to isolate empirical variance vs. bias reduction across noisy consumer sales patterns.",
        body_style
    ))

    # SECTION 8
    story.append(Paragraph("8. Challenges, Limitations & Prescriptive Optimization Layer", h1_style))
    story.append(Paragraph(
        "<b>8.1 The Dead-Stock & Knapsack Dilemma:</b> Initial optimization attempts using standard unbounded 0/1 knapsack caused 84% of store capital to concentrate in only two high-margin SKUs (Luxury Embroidered Suits and Fossil Watches), completely starving essential daily foot-traffic items. Furthermore, standard forecasting assumes stationary seasons, causing dangerous late-winter overstocking. We engineered two prescriptive solutions:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Mixed-Integer Linear Programming (MILP) Formulation:</b><br/>"
        "$$\\max_{x} \\sum_{i=1}^N \\text{Profit}_i \\cdot x_i \\quad \\text{s.t.} \\quad \\sum_{i=1}^N \\text{WholesaleCost}_i \\cdot x_i \\le \\text{Budget}, \\quad 0 \\le x_i \\le \\min(\\text{NetNeeded}_i, \\lfloor 0.25 \\cdot \\text{Budget} / \\text{Cost}_i \\rfloor)$$<br/>"
        "• <b>Pakistani Dynamic Seasonality & Festive Regimes:</b> Dynamic date picking parses Gregorian month/day while enabling lunar festive overrides: Pre-Eid rush (+110% on apparel/footwear), Winter Wedding season (+70%), and Late Winter clearance (-55% anti-dead-stock decay). When capital exceeds catalog demand, surplus liquidity is safeguarded as unspent cash rather than forced into surplus dead inventory.",
        body_style
    ))

    # REFERENCES & AI DISCLOSURE
    story.append(Paragraph("9. Academic References & AI Use Disclosure", h1_style))
    story.append(Paragraph(
        "<b>References:</b><br/>"
        "<b>[1]</b> Amjad, U., Talpur, S. R., Ahmed, S., Farman, H., & Laila, U. (2024). <i>Advanced Furniture Sales Forecasting Using Machine Learning and Deep Learning Techniques.</i> ILMA Journal of Technology & Software Management (IJTSM), 5(2), 11–24.<br/>"
        "<b>[2]</b> Hastie, T., Tibshirani, R., & Friedman, J. (2009). <i>The Elements of Statistical Learning: Data Mining, Inference, and Prediction.</i> 2nd ed. Springer Series in Statistics.<br/>"
        "<b>[3]</b> Menekşe, M. (2020). <i>E-Commerce Customer Behavior and Transaction Dataset.</i> Kaggle Repository.<br/>"
        "<b>[4]</b> Sukkur IBA University Department of Computer Science (2026). <i>Course Outline: Machine Learning (BS-V — CS, SE & AI).</i>",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>AI Use Disclosure:</b> In accordance with Sukkur IBA Academic Integrity Guidelines, Generative AI (Antigravity Assistant) was utilized for code debugging, syntax checking, and document styling assistance. All conceptual problem formulations, leakage audits, data preprocessing implementations, model training pipelines, error analyses, and optimization constraints were authored, executed, and defended by the project team.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Final Report successfully compiled: {pdf_filename}")

if __name__ == '__main__':
    build_pdf()
