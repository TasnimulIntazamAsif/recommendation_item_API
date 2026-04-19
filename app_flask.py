from flask import Flask, request, jsonify
from flasgger import Swagger
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
swagger = Swagger(app)

# ==========================================
# 1. মডেল এবং ডেটা লোড করা
# ==========================================
print("Loading context-aware model and catalog...")
# নিশ্চিত করুন আপনার সেভ করা মডেলের পাথ সঠিক আছে
loaded_model = tf.saved_model.load("saved_model/attention_wide_deep")
infer = loaded_model.signatures["serving_default"]

# ক্যাটালগ লোড (আইটেম আইডি থেকে নাম এবং লেবেল পাওয়ার জন্য)
df_full = pd.read_csv("data/large_dataset_chronological.csv")
df_items = df_full.drop_duplicates(subset=["itemId"])

ITEM_CATALOG = dict(zip(df_items.itemId, df_items.itemName))
ITEM_LABEL_MAP = dict(zip(df_items.itemId, df_items.itemLabel))
CANDIDATE_IDS = list(ITEM_CATALOG.keys())

FESTIVAL_DATES = ["21-02", "26-03", "14-04", "01-05", "16-12", "25-12", "31-12"]


# ==========================================
# 2. কন্টেক্সট এক্সট্রাকশন ফাংশন
# ==========================================
def extract_context(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        dt = datetime.now()

    is_month_start = 1.0 if dt.day <= 10 else 0.0
    slot = "Morning" if 5 <= dt.hour < 12 else "Afternoon" if 12 <= dt.hour < 18 else "Evening" if 18 <= dt.hour < 20 else "Night"
    season = "Winter" if dt.month in [11, 12, 1, 2] else "Summer" if dt.month in [3, 4, 5] else "Monsoon"
    is_holiday = 1.0 if dt.weekday() in [4, 5] else 0.0
    is_festival = 1.0 if dt.strftime("%d-%m") in FESTIVAL_DATES else 0.0

    return season, slot, is_holiday, is_festival, is_month_start


# ==========================================
# 3. রেকমেন্ডেশন এন্ডপয়েন্ট
# ==========================================
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Context-Aware Item Recommendation API
    ---
    tags:
      - Recommendations
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: RecRequest
          properties:
            customerid:
              type: integer
              example: 23384
            date:
              type: string
              example: "2026-04-14 18:30:00"
            items:
              type: array
              items:
                type: object
                properties:
                  itemid:
                    type: integer
                  quantity:
                    type: integer
    responses:
      200:
        description: List of purchased and recommended items.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    cust_id = data.get("customerid")
    order_date = data.get("date")

    # মাল্টিপল আইটেম হ্যান্ডলিং
    input_items = data.get("items", [])
    cart_item_ids = [item['itemid'] for item in input_items]

    # কন্টেক্সট বের করা
    season, slot, is_holiday, is_festival, is_month_start = extract_context(order_date)

    batch_size = len(CANDIDATE_IDS)
    candidate_labels = [ITEM_LABEL_MAP.get(cid, "General Item") for cid in CANDIDATE_IDS]

    # ✅ মডেল ইনপুট টেনসর তৈরি
    inputs = {
        "customerId": tf.convert_to_tensor(np.array([cust_id] * batch_size, dtype=np.int64).reshape(-1, 1)),
        "itemId": tf.convert_to_tensor(np.array(CANDIDATE_IDS, dtype=np.int64).reshape(-1, 1)),
        "itemLabel": tf.convert_to_tensor(np.array(candidate_labels, dtype=object).reshape(-1, 1)),
        "season": tf.convert_to_tensor(np.array([season] * batch_size, dtype=object).reshape(-1, 1)),
        "timeSlot": tf.convert_to_tensor(np.array([slot] * batch_size, dtype=object).reshape(-1, 1)),
        "isHoliday": tf.convert_to_tensor(np.array([is_holiday] * batch_size, dtype=np.float32).reshape(-1, 1)),
        "isFestival": tf.convert_to_tensor(np.array([is_festival] * batch_size, dtype=np.float32).reshape(-1, 1)),
        "is_month_start": tf.convert_to_tensor(
            np.array([is_month_start] * batch_size, dtype=np.float32).reshape(-1, 1)),
        "quantity": tf.convert_to_tensor(np.array([1.0] * batch_size, dtype=np.float32).reshape(-1, 1))
    }

    # ✅ মডেল প্রেডিকশন
    preds_dict = infer(**inputs)
    output_key = list(preds_dict.keys())[0]
    preds = preds_dict[output_key].numpy()

    rec_list = []
    for i, cid in enumerate(CANDIDATE_IDS):
        # অলরেডি কেনা আইটেমগুলো রেকমেন্ডেশন থেকে বাদ দেওয়া হচ্ছে
        if cid in cart_item_ids: continue

        base_score = float(preds[i][0])
        boost = 1.0
        current_label = candidate_labels[i]

        # ডাইনামিক ক্যাটাগরি বুস্টিং (মাল্টিপল ইনপুট আইটেমের ওপর ভিত্তি করে)
        if cart_item_ids:
            for cart_id in cart_item_ids:
                if current_label == ITEM_LABEL_MAP.get(cart_id, "General"):
                    boost += 0.15  # একই ক্যাটাগরির আইটেম হলে ১৫% বুস্ট

        final_score = min((base_score * boost) * 100, 100.0)

        rec_list.append({
            "itemid": int(cid),
            "item_name": ITEM_CATALOG.get(cid, "Unknown"),
            "score": round(final_score, 2)
        })

    # স্কোর অনুযায়ী সর্ট করে টপ ৫ নেওয়া
    sorted_recs = sorted(rec_list, key=lambda x: x['score'], reverse=True)[:5]

    # কেনা আইটেমগুলোর নাম বের করা
    purchased_names = [ITEM_CATALOG.get(cid, "Unknown") for cid in cart_item_ids]

    # ✅ ক্লিন আউটপুট রিটার্ন
    return jsonify({
        "purchased_items": purchased_names,
        "recommended_items": sorted_recs
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)