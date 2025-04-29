import streamlit as st
import random
import time
import matplotlib.pyplot as plt

# --------------------
# Load Questions
# --------------------
from questions.questions import questions  # Import your list of questions

# --------------------
# Streamlit Setup
# --------------------
st.set_page_config(page_title="Generative AI Quiz", layout="wide")
st.title("Databricks Gen AI Practice Quiz")

# --------------------
# Initialize session state
# --------------------
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'selected_questions' not in st.session_state:
    st.session_state.selected_questions = random.sample(questions, min(45, len(questions)))
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}

# --------------------
# Start Quiz Button
# --------------------
if not st.session_state.quiz_started:
    if st.button("Start Quiz ✨"):
        st.session_state.quiz_started = True
        st.session_state.start_time = time.time()
    st.stop()

# --------------------
# Display Questions Normally
# --------------------
for idx, q in enumerate(st.session_state.selected_questions):
    st.markdown(f"#### Q{idx+1}. {q['question']}")
    selected = st.radio(
        f"Choose your answer for Q{idx+1}:",
        q["options"],
        key=f"question_{idx}",
        index=None,
        label_visibility="collapsed"
    )
    st.session_state.user_answers[idx] = selected
    st.markdown("---")

# --------------------
# Submit All Button
# --------------------
if st.button("Submit All Answers ✅"):
    end_time = time.time()
    total_time = end_time - st.session_state.start_time

    correct = 0
    wrong = 0
    results = []

    for idx, q in enumerate(st.session_state.selected_questions):
        user_ans = st.session_state.user_answers.get(idx)
        correct_ans = q['answer']
        is_correct = user_ans == correct_ans

        if is_correct:
            correct += 1
        else:
            wrong += 1

        results.append({
            "question": q['question'],
            "user_answer": user_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct
        })

    # --------------------
    # Dashboard View
    # --------------------
    st.divider()
    st.header("🚀 Final Score Dashboard")
    st.divider()

    percentage = (correct / 45) * 100

    # Define emoji
    if correct > 35:
        emoji = "😀"
    elif correct > 20:
        emoji = "😐"
    elif correct > 10:
        emoji = "😞"
    else:
        emoji = "😭"

    # Create donut chart
    fig, ax = plt.subplots(figsize=(3, 3))  # Made a bit smaller
    size = 0.3

    if percentage >= 75:
        color = '#7ED957'  # green
    elif percentage >= 50:
        color = '#FFBD59'  # yellow-orange
    else:
        color = '#FF595E'  # red

    ax.pie(
        [correct, 45 - correct],
        radius=1,
        colors=[color, '#e0e0e0'],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=size, edgecolor='white')
    )

    ax.text(0, 0.1, f"{int(percentage)}%", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0, -0.2, "SCORE", ha='center', va='center', fontsize=10)
    ax.text(0, -0.4, emoji, ha='center', va='center', fontsize=20)

    ax.set(aspect="equal")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(fig)

    with col2:
        st.markdown("## 🎯 Your Score")
        st.markdown(f"### **{correct} / 45 **")
        st.markdown("---")
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        st.markdown(f"⏱️ **Time Taken:** {minutes} min {seconds} sec")

    # --------------------
    # Show Detailed Review
    # --------------------
    st.divider()
    st.header("📋 Review Your Answers")  # Full Width Now!
    st.divider()

    for idx, res in enumerate(results):
        st.markdown(f"### Q{idx+1}: {'✅ Correct' if res['is_correct'] else '❌ Wrong'}")
        st.markdown(f"**Question:** {res['question']}")
        st.markdown(f"**Your Answer:** {res['user_answer']}")
        st.markdown(f"**Correct Answer:** {res['correct_answer']}")
        st.markdown("---")

    st.button("Restart Quiz 🔄", on_click=lambda: st.session_state.clear())
