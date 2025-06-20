# Flask API Integration

This API provides two main endpoints for ML model prediction and report generation.

---

## Endpoints

### POST `/predict`

**Description**:  
Predicts the type of a request based on the provided text using a pre-trained ML model.

**Request Body** (JSON):

```json
{
  "request_text": "Your request text here",
  "request_type": "Optional: existing type if any"
}
```

**Response** (JSON):

```json
{
  "prediction": "suggested_type_of_the_request"
}
```

---

### POST `/report`

**Description**:  
Generates a report based on the provided data.

**Request Body** (JSON):  
Expects a JSON object containing the data for report generation.

**Response (Success)**:

```json
{
  "status": "ok",
  "graphs": [
    // JSON representation of generated graphs
  ]
}
```

**Response (Error)**:

```json
{
  "error": "Error message"
}
```
