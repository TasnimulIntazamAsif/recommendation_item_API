# recommendation_item_API

Retail recommendation system-এর জন্য এটি একটি Flask API project, যেখানে offline training artifacts ব্যবহার করে real-time এ top-N item recommendation generate করা হয়।

## Project Overview (First to Last)

এই project-এর end-to-end flow মূলত 4টা stage-এ কাজ করে:

1. **Data & Feature Preparation (Notebook Stage)**  
   - Raw transaction/basket data থেকে category mapping, embeddings, co-purchase pattern, user history signals তৈরি করা হয়।
   - এই processing ধাপগুলো notebooks-এ করা হয়েছে (যেমন stage artifact build, dataset creation, retraining notebooks)।

2. **Model Training (Ranker Stage 2)**  
   - Stage-1 থেকে পাওয়া candidate/feature signals ব্যবহার করে `XGBoost Ranker` train করা হয়।
   - Trained model এবং feature schema export করা হয়:
     - `xgboost_ranker_model.json`
     - `ranker_feature_columns.json`
     - `training_summary.json`
     - `final_result_summary.csv`

3. **Artifact Packaging (Serving Assets_stage 1)**  
   - Inference এর জন্য model এর পাশাপাশি কয়েকটি precomputed lookup/pickle assets রাখা হয়, যেমন:
     - association rules
     - item pair counter (co-purchase)
     - context-item counter
     - user-item / user-category counters
     - item/category/name mapping
     - category embedding vectors
   - এগুলো `model_assets` structure-এর মধ্যে load-ready format এ থাকে।

4. **Online Inference API (Flask Serving Stage)**  
   - Client `/api/recommend` endpoint এ `customerid`, `date`, এবং cart `items` পাঠায়।
   - API candidate generation + ranking + business reranking apply করে final recommendation return করে।

## Runtime Recommendation Pipeline

`/api/recommend` call হলে সার্ভিসে নিচের ধারাবাহিক flow execute হয়:

1. **Request context build**  
   - Date থেকে season/holiday/festival/weekOfMonth বের করা হয়  
   - Customer-এর default timeslot যুক্ত করা হয়

2. **Stage-1 candidate generation**  
   - Association rules  
   - Co-purchase signal  
   - Context popularity signal  
   - User purchase history signal  
   - Weighted merge করে top candidate pool তৈরি

3. **Stage-2 scoring feature build**  
   - basket_size, cooccurrence, affinity, context popularity, history, embedding similarity, business boost ইত্যাদি feature বানানো হয়
   - categorical feature one-hot করে trained feature columns এর সাথে align করা হয়

4. **Ranker prediction**  
   - XGBoost Ranker candidate-level score predict করে

5. **Business logic reranking**  
   - Meal intent detect (cooking/breakfast/snack/generic)
   - hard category filter, category boost, category coverage, exact category cap apply
   - final score normalize করে user-facing score বানানো হয়

6. **Final response build**  
   - Input items (name + quantity) এবং ranked recommendations (rank, score, raw_score, category) return করা হয়

## API Endpoints



- `POST /api/recommend`  
  Top recommendation result return করে

Example request payload:

```json
{
  "customerid": 23412,
  "date": "2026-04-20",
  "items": [
    { "itemid": 13989, "quantity": 1 }
  ]
}
```

## Project Structure (Key Files)

- `app.py` - Flask app run entrypoint  
- `app/__init__.py` - app factory + blueprint registration  
- `app/routes/recommendation_routes.py` - API routes (`health`, `metrics`, `recommend`)  
- `app/services/recommender_service.py` - core recommendation pipeline logic  
- `app/utils/helpers.py` - shared helper functions  
- `config.py` - model/artifact/data path configuration  
- `model_assets/` - model + stage artifacts + outputs

## Run Locally

1. Python dependencies install করুন (project environment অনুযায়ী)
2. নিশ্চিত করুন `config.py` path অনুযায়ী model assets available আছে
3. Server run করুন:

```bash
python app.py
```

4. Swagger/API docs browse করুন (Flasgger enabled থাকলে):
   - `http://localhost:5000/apidocs`