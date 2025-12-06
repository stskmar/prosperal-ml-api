"""
FastAPI Microservice for Bank Lead Scoring

Microservice ini menyediakan REST API untuk inference model lead scoring.

Endpoints:
- GET /health: Health check
- POST /score: Lead scoring inference (single record)
- POST /bulk-score: Bulk lead scoring inference (CSV upload, max 1000 rows)

Run server:
    uvicorn main:app --reload --port 8000
    
Test endpoints:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d @sample_request.json
    curl -X POST http://localhost:8000/bulk-score -F "file=@leads.csv"
"""

import logging
import io
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from inference_service import (
    load_artifacts,
    run_inference,
    InferenceError,
    Artifacts,
    # Bulk inference imports
    MAX_BULK_ROWS,
    REQUIRED_INPUT_FIELDS,
    validate_dataframe_columns,
    clean_dataframe,
    run_bulk_inference,
)


# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN EVENT HANDLER
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler untuk startup dan shutdown."""
    # Startup
    global artifacts
    logger.info("=" * 60)
    logger.info("Starting Bank Lead Scoring Service")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading model artifacts...")
        artifacts = load_artifacts("./artifacts")
        logger.info(f"✓ Artifacts loaded successfully")
        logger.info(f"✓ Model features: {len(artifacts.feature_names)}")
        logger.info(f"✓ Feature names: {artifacts.feature_names}")
        logger.info("=" * 60)
        logger.info("Service ready to accept requests")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"✗ Failed to load artifacts: {e}")
        logger.error("Service cannot start without model artifacts")
        raise RuntimeError("Cannot start service without model artifacts")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Bank Lead Scoring Service")


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================
app = FastAPI(
    title="Bank Lead Scoring API",
    description="Microservice untuk inference model lead scoring bank",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (optional, uncomment jika diperlukan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # Next.js frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# =============================================================================
# GLOBAL STATE - MODEL ARTIFACTS
# =============================================================================
artifacts: Artifacts = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class LeadFeatures(BaseModel):
    """Request model untuk customer features."""
    age: int = Field(..., ge=18, le=100, description="Usia nasabah")
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ] = Field(..., description="Jenis pekerjaan")
    marital: Literal["divorced", "married", "single"] = Field(..., description="Status pernikahan")
    education: Literal["primary", "secondary", "tertiary", "unknown"] = Field(..., description="Tingkat pendidikan")
    default: Literal["no", "yes"] = Field(..., description="Apakah pernah default kredit")
    balance: int = Field(..., description="Saldo tahunan rata-rata (dalam EUR)")
    housing: Literal["no", "yes"] = Field(..., description="Apakah memiliki kredit rumah")
    loan: Literal["no", "yes"] = Field(..., description="Apakah memiliki pinjaman personal")
    contact: Literal["cellular", "telephone", "unknown"] = Field(..., description="Tipe komunikasi kontak")
    day: int = Field(..., ge=1, le=31, description="Hari terakhir kontak dalam bulan")
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"] = Field(
        ..., description="Bulan terakhir kontak"
    )
    campaign: int = Field(..., ge=1, description="Jumlah kontak selama kampanye ini")
    pdays: int = Field(..., ge=-1, description="Hari sejak kontak terakhir (-1 = belum pernah)")
    previous: int = Field(..., ge=0, description="Jumlah kontak sebelum kampanye ini")
    poutcome: Literal["failure", "other", "success", "unknown"] = Field(
        ..., description="Hasil kampanye marketing sebelumnya"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }
    )


class ReasonCode(BaseModel):
    """Model untuk single reason code."""
    feature: str = Field(..., description="Nama fitur yang berpengaruh")
    direction: Literal["positive", "negative"] = Field(..., description="Arah kontribusi")
    shap_value: float = Field(..., description="Nilai SHAP (importance)")


class InferenceResult(BaseModel):
    """Response model untuk hasil inference."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilitas subscribe (0.0 - 1.0)")
    prediction: Literal[0, 1] = Field(..., description="Prediksi binary (0 = tidak, 1 = ya)")
    prediction_label: Literal["yes", "no"] = Field(..., description="Label prediksi")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Level konversi")
    reason_codes: List[ReasonCode] = Field(
        ..., min_length=5, max_length=5, description="Top 5 fitur paling berpengaruh"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "probability": 0.128,
                "prediction": 0,
                "prediction_label": "no",
                "risk_level": "Low",
                "reason_codes": [
                    {"feature": "contact", "direction": "positive", "shap_value": 0.277},
                    {"feature": "housing", "direction": "negative", "shap_value": -0.229},
                    {"feature": "day", "direction": "positive", "shap_value": 0.177},
                    {"feature": "balance", "direction": "positive", "shap_value": 0.168},
                    {"feature": "month", "direction": "negative", "shap_value": -0.159}
                ]
            }
        }
    )


class HealthResponse(BaseModel):
    """Response model untuk health check."""
    status: str = Field(..., description="Status service")
    model_loaded: bool = Field(..., description="Apakah model sudah di-load")
    feature_count: int = Field(..., description="Jumlah fitur model")


# =============================================================================
# BULK INFERENCE PYDANTIC MODELS
# =============================================================================

class InvalidRow(BaseModel):
    """Model untuk baris yang tidak valid/di-drop."""
    row_index: int = Field(..., description="Index baris asli dalam CSV (0-based)")
    reason: str = Field(..., description="Alasan baris di-drop")


class BulkReasonCode(BaseModel):
    """Model untuk single reason code dalam bulk response."""
    feature: str = Field(..., description="Nama fitur yang berpengaruh")
    direction: Literal["positive", "negative"] = Field(..., description="Arah kontribusi")
    shap_value: float = Field(..., description="Nilai SHAP (importance)")


class BulkPrediction(BaseModel):
    """Model untuk single prediction dalam bulk response."""
    row_index: int = Field(..., description="Index baris asli dalam CSV (0-based)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilitas subscribe")
    prediction: Literal[0, 1] = Field(..., description="Prediksi binary")
    prediction_label: Literal["yes", "no"] = Field(..., description="Label prediksi")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Level konversi")
    reason_codes: List[BulkReasonCode] = Field(..., description="Top 5 fitur paling berpengaruh")


class BulkSummary(BaseModel):
    """Model untuk summary statistik bulk inference."""
    total_rows: int = Field(..., description="Total baris dalam CSV")
    processed_rows: int = Field(..., description="Jumlah baris yang berhasil diproses")
    dropped_rows: int = Field(..., description="Jumlah baris yang di-drop")
    avg_probability: Optional[float] = Field(None, description="Rata-rata probability")
    conversion_high: int = Field(..., description="Jumlah High conversion")
    conversion_medium: int = Field(..., description="Jumlah Medium conversion")
    conversion_low: int = Field(..., description="Jumlah Low conversion")


class BulkScoreResponse(BaseModel):
    """Response model untuk bulk scoring."""
    success: bool = Field(..., description="Apakah request berhasil")
    summary: BulkSummary = Field(..., description="Summary statistik")
    invalid_rows: List[InvalidRow] = Field(default=[], description="Daftar baris yang tidak valid")
    predictions: List[BulkPrediction] = Field(default=[], description="Hasil prediksi per baris")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "summary": {
                    "total_rows": 100,
                    "processed_rows": 95,
                    "dropped_rows": 5,
                    "avg_probability": 0.234,
                    "conversion_high": 15,
                    "conversion_medium": 30,
                    "conversion_low": 50
                },
                "invalid_rows": [
                    {"row_index": 5, "reason": "missing_values: age, balance"},
                    {"row_index": 12, "reason": "invalid_numeric: balance"}
                ],
                "predictions": [
                    {
                        "row_index": 0,
                        "probability": 0.78,
                        "prediction": 1,
                        "prediction_label": "yes",
                        "risk_level": "High",
                        "reason_codes": [
                            {"feature": "poutcome", "direction": "positive", "shap_value": 0.45},
                            {"feature": "contact", "direction": "positive", "shap_value": 0.28},
                            {"feature": "housing", "direction": "negative", "shap_value": -0.22},
                            {"feature": "balance", "direction": "positive", "shap_value": 0.17},
                            {"feature": "month", "direction": "negative", "shap_value": -0.15}
                        ]
                    }
                ]
            }
        }
    )


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError (input validation errors)."""
    logger.warning(f"Invalid input: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc)
        }
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    """Handle InferenceError (model inference errors)."""
    logger.error(f"Inference error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Inference failed",
            "detail": "An error occurred during model inference"
        }
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Bank Lead Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "score": "/score",
            "bulk_score": "/bulk-score",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status dan informasi model.
    """
    return HealthResponse(
        status="ok",
        model_loaded=artifacts is not None,
        feature_count=len(artifacts.feature_names) if artifacts else 0
    )


@app.post("/score", response_model=InferenceResult, tags=["Inference"])
async def score_lead(features: LeadFeatures):
    """
    Lead scoring inference endpoint.
    
    Menerima customer features dan mengembalikan:
    - Probability: Probabilitas customer akan subscribe
    - Prediction: Binary prediction (0 atau 1)
    - Risk level: Low/Medium/High
    - Reason codes: Top 5 fitur yang paling berpengaruh
    
    Parameters
    ----------
    features : LeadFeatures
        Customer features untuk di-score
    
    Returns
    -------
    InferenceResult
        Hasil inference dengan probability, prediction, dan reason codes
    
    Raises
    ------
    400
        Jika input tidak valid
    500
        Jika terjadi error saat inference
    """
    # Check if model loaded
    if artifacts is None:
        raise InferenceError("Model artifacts not loaded")
    
    # Convert Pydantic model to dict
    input_dict = features.model_dump()
    
    logger.info(f"Processing inference request for customer: age={features.age}, job={features.job}")
    
    # Run inference
    result = run_inference(input_dict, artifacts)
    
    logger.info(f"Inference complete: probability={result['probability']:.4f}, prediction={result['prediction_label']}")
    
    return InferenceResult(**result)


@app.post("/bulk-score", response_model=BulkScoreResponse, tags=["Inference"])
async def bulk_score_leads(file: UploadFile = File(..., description="CSV file containing leads data")):
    """
    Bulk lead scoring inference endpoint.
    
    Menerima file CSV berisi multiple customer records dan mengembalikan
    hasil scoring untuk setiap baris yang valid.
    
    Parameters
    ----------
    file : UploadFile
        CSV file dengan kolom yang sesuai dengan REQUIRED_COLUMNS
    
    Returns
    -------
    BulkScoreResponse
        Hasil inference untuk semua baris valid, plus summary dan invalid rows
    
    Raises
    ------
    400
        - File bukan CSV
        - File kosong
        - Melebihi MAX_BULK_ROWS (1000 baris)
        - Missing required columns
    500
        Jika terjadi error saat inference
    
    Notes
    -----
    - Max 1000 rows per request
    - Baris dengan missing values akan di-drop dan dilaporkan di invalid_rows
    - Kolom yang tidak dikenal akan diabaikan (hanya log warning)
    """
    # Check if model loaded
    if artifacts is None:
        raise InferenceError("Model artifacts not loaded")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "message": "Only CSV files are accepted",
                "filename": file.filename
            }
        )
    
    logger.info(f"Processing bulk inference request: {file.filename}")
    
    try:
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty file",
                    "message": "Uploaded CSV file is empty"
                }
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "CSV parsing error",
                    "message": f"Failed to parse CSV: {str(e)}"
                }
            )
        
        total_rows = len(df)
        logger.info(f"CSV loaded: {total_rows} rows, {len(df.columns)} columns")
        
        # Validate row count
        if total_rows == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty CSV",
                    "message": "CSV file contains no data rows"
                }
            )
        
        if total_rows > MAX_BULK_ROWS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Too many rows",
                    "message": f"CSV contains {total_rows} rows, maximum allowed is {MAX_BULK_ROWS}",
                    "max_rows": MAX_BULK_ROWS
                }
            )
        
        # Validate columns
        validation_result = validate_dataframe_columns(df)
        missing_cols = validation_result["missing_columns"]
        extra_cols = validation_result["extra_columns"]
        
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required columns",
                    "message": f"CSV is missing required columns: {missing_cols}",
                    "missing_columns": missing_cols,
                    "required_columns": REQUIRED_INPUT_FIELDS
                }
            )
        
        if extra_cols:
            logger.warning(f"CSV contains extra columns that will be ignored: {extra_cols}")
        
        # Clean dataframe (handle missing values and invalid data)
        clean_result = clean_dataframe(df)
        cleaned_df = clean_result["cleaned_df"]
        original_indices = clean_result["original_indices"]
        dropped_indices = clean_result["dropped_indices"]
        invalid_rows_data = clean_result["invalid_rows"]
        
        processed_rows = len(cleaned_df)
        dropped_rows = len(dropped_indices)
        
        logger.info(f"Data cleaning complete: {processed_rows} valid rows, {dropped_rows} dropped")
        
        # Build invalid rows response - combine dropped (missing values) with invalid (parsing errors)
        invalid_rows_response = []
        
        # Add dropped rows (missing values)
        for idx in dropped_indices:
            # Find which columns had missing values
            row = df.iloc[idx]
            missing_cols_in_row = [col for col in REQUIRED_INPUT_FIELDS if pd.isnull(row.get(col))]
            invalid_rows_response.append(
                InvalidRow(row_index=int(idx), reason=f"missing_values: {', '.join(missing_cols_in_row)}")
            )
        
        # Add invalid rows (parsing errors)
        for row_info in invalid_rows_data:
            invalid_rows_response.append(
                InvalidRow(row_index=row_info["row_index"], reason=row_info["reason"])
            )
        
        # If no valid rows after cleaning
        if processed_rows == 0:
            return BulkScoreResponse(
                success=True,
                summary=BulkSummary(
                    total_rows=total_rows,
                    processed_rows=0,
                    dropped_rows=dropped_rows,
                    avg_probability=None,
                    conversion_high=0,
                    conversion_medium=0,
                    conversion_low=0
                ),
                invalid_rows=invalid_rows_response,
                predictions=[]
            )
        
        # Run bulk inference (original_indices already from clean_result)
        results = run_bulk_inference(cleaned_df, original_indices, artifacts)
        
        # Build predictions response
        predictions_response = []
        conversion_counts = {"High": 0, "Medium": 0, "Low": 0}
        total_probability = 0.0
        
        for result in results:
            # Convert reason codes
            reason_codes = [
                BulkReasonCode(
                    feature=rc["feature"],
                    direction=rc["direction"],
                    shap_value=rc["shap_value"]
                )
                for rc in result["reason_codes"]
            ]
            
            prediction = BulkPrediction(
                row_index=result["row_index"],
                probability=result["probability"],
                prediction=result["prediction"],
                prediction_label=result["prediction_label"],
                risk_level=result["risk_level"],
                reason_codes=reason_codes
            )
            predictions_response.append(prediction)
            
            # Update stats
            conversion_counts[result["risk_level"]] += 1
            total_probability += result["probability"]
        
        avg_probability = total_probability / processed_rows if processed_rows > 0 else None
        
        logger.info(
            f"Bulk inference complete: "
            f"High={conversion_counts['High']}, "
            f"Medium={conversion_counts['Medium']}, "
            f"Low={conversion_counts['Low']}, "
            f"avg_prob={avg_probability:.4f}" if avg_probability else ""
        )
        
        return BulkScoreResponse(
            success=True,
            summary=BulkSummary(
                total_rows=total_rows,
                processed_rows=processed_rows,
                dropped_rows=dropped_rows,
                avg_probability=round(avg_probability, 4) if avg_probability else None,
                conversion_high=conversion_counts["High"],
                conversion_medium=conversion_counts["Medium"],
                conversion_low=conversion_counts["Low"]
            ),
            invalid_rows=invalid_rows_response,
            predictions=predictions_response
        )
        
    except HTTPException:
        raise
    except InferenceError:
        raise
    except Exception as e:
        logger.error(f"Bulk inference error: {e}", exc_info=True)
        raise InferenceError(f"Bulk inference failed: {str(e)}")


# =============================================================================
# MAIN (for local development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
