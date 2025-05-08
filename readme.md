# Classroom Simulation with LLM Agents with LangChain and LangGraph

A self-contained Python script that uses **LangChain** and **LangGraph** to simulate a realistic high-school math class. Three agent types interact in a fixed number of “rounds”:

1. **ModeratorAgent** picks who speaks next  
2. **StudentAgent** generates a short student reply  
3. **TeacherAgent** responds and applies negative punishment if needed  

Every message is capped at 20 words to preserve the feel of a natural classroom dialogue.

---

## 🗂️ Overview of Core Classes

### `SimulationState`
- **Purpose**: Holds the entire simulation’s mutable state.
- **Key fields**:
  - `students` – list of all student agents  
  - `teacher` – the teacher agent  
  - `moderator` – the moderator agent  
  - `rounds` – total number of conversation turns  
  - `lesson_plan` – the text describing the lesson’s goals, tone, and rules  
  - `current_round` – which turn we’re on  
  - `selected_index` – index of the student chosen this round  
  - `last_message` – last utterance in the dialogue  
- **Tip**: You can add extra fields—e.g. `class_mood` or `strikes_remaining`—to model more complex dynamics!

---

### `StudentAgent`
- **Purpose**: Emulates a single student’s personality and turn-taking.
- **Constructor arguments**:
  - `name` (str): student’s name  
  - `age` (int): age, used to shape tone  
  - `profile` (str): short bullet-point profile, injected into every prompt  
  - `base_prompt` (str): the LLM template describing how this student thinks and speaks
- **Method**:
  - `act(last_teacher_message, state)` → `str`  
    - Builds a dynamic prompt by combining `base_prompt`, the student’s `profile`, their own `history`, the last teacher message, and the shared `lesson_plan`.  
    - Invokes the LLM and appends the reply to this student’s history.
- **Customization ideas**:
  - **Add new traits**: e.g. a “shy” student who hesitates or a “competitive” student who tries to outperform peers.  
  - **Adjust verbosity**: change max word count in the prompt.  
  - **Track metrics**: add `self.participation_count` or `self.correct_answers` to reward or penalize.

---

### `TeacherAgent`
- **Purpose**: Guides the lesson, maintains discipline, and replies to student comments.
- **Constructor arguments**:
  - `name` (str), `subject` (str), `methodology` (str), `base_prompt` (str)  
    - `methodology` is used for internal documentation but can also be injected into the prompt if you want more nuanced punishments.
- **Method**:
  - `respond(student_message, student, state)` → `str`  
    - Crafts an LLM prompt combining the teacher’s `base_prompt`, the student’s name and last message, and the `lesson_plan`.  
    - Returns an authoritative but natural response
- **Customization ideas**:
  - **Vary disciplinary style**: switch from “negative punishment” to “positive reinforcement” by editing the `methodology` and teacher prompt.  
  - **Dynamic feedback**: track `state['misbehavior_count']` and escalate consequences over time.

---

### `ModeratorAgent`
- **Purpose**: Replaces a random choice with an LLM-driven decision on which student should speak next.
- **Constructor arguments**:
  - `base_prompt` (str): describes the moderator’s role and decision criteria.
- **Method**:
  - `select(students, last_message, state)` → `int`  
    - Builds a list of student names and profiles, then asks the LLM to choose the best candidate based on the recent teacher message and the lesson plan.  
    - Falls back to random if the LLM reply isn’t a valid index.
- **Customization ideas**:
  - **Weighted selection**: maintain a `participation_score` per student in `state` and bias the selection prompt.  
  - **Add criteria**: e.g. “avoid picking the same student twice in a row” by injecting extra constraints.

---

## ⚙️ How It Works

1. **Initialize**  
   - Set up your API key and model with  
     ```python
     llm = init_chat_model("llama3-8b-8192", model_provider="groq")
     ```  
   - Or swap in another provider/model:  
     ```python
     llm = init_chat_model("gpt-4o-mini", model_provider="openai")
     ```
2. **`init_state()`**  
   - Defines `rounds`, your `lesson_plan` text, and the initial welcome message.  
   - Perfect spot to **tweak** the number of rounds or rewrite the entire lesson narrative!
3. **Workflow Graph**  
   - `select_student` → `student_act` → `teacher_respond` → `increment_round`  
   - Loops until `current_round == rounds`
4. **Run**  
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   python3 students-simulator.py
