import os
import random
from typing import List, Any, TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# API configuration
api_key = os.getenv("GROQ_API_KEY", "[USE_YOUR_API_KEY]")
os.environ["GROQ_API_KEY"] = api_key
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

class SimulationState(TypedDict, total=False):
    students: List[Any]
    teacher: Any
    moderator: Any
    rounds: int
    lesson_plan: str
    current_round: int
    selected_index: int
    last_message: str

class StudentAgent:
    def __init__(self, name: str, age: int, profile: str, base_prompt: str):
        self.name = name
        self.age = age
        self.profile = profile
        self.base_prompt = base_prompt
        self.history: List[str] = []

    def act(self, teacher_message: str, state: SimulationState) -> str:
        context = (
            f"Profile: {self.profile}. History: {self.history}. "
            f"Last teacher message: {teacher_message}. Lesson plan: {state['lesson_plan']}"
        )
        prompt = (
            f"{self.base_prompt}\n"
            f"Context: You are in a math class; the teacher said: \"{teacher_message}\". "
            f"You are {self.name}, a {self.age}-year-old student. Speak naturally, like a real student. "
            f"Remember: you are in class, other classmates are listening, and the atmosphere is typical of a high school classroom. "
            f"Your response should sound believable and be brief (no more than 20 words)."
        )
        response = llm.invoke(prompt).content
        self.history.append(response)
        return response

class TeacherAgent:
    def __init__(self, name: str, subject: str, methodology: str, base_prompt: str):
        self.name = name
        self.subject = subject
        self.methodology = methodology
        self.base_prompt = base_prompt

    def respond(self, student_message: str, student: StudentAgent, state: SimulationState) -> str:
        prompt = (
            f"{self.base_prompt}\n"
            f"You are in an actual classroom. Student {student.name} said: \"{student_message}\". "
            f"Respond as the teacher: authoritative yet natural, in no more than 20 words. "
            f"Avoid sounding scripted or artificial. Your goal is to keep the class focused and enforce boundaries clearly. "
            f"Lesson plan: {state['lesson_plan']}"
        )
        return llm.invoke(prompt).content

class ModeratorAgent:
    def __init__(self, base_prompt: str):
        self.base_prompt = base_prompt

    def select(self, students: List[StudentAgent], last_message: str, state: SimulationState) -> int:
        student_list = "".join([
            f"- {i}: {student.name}, profile: {student.profile}\n"
            for i, student in enumerate(students)
        ])
        prompt = (
            f"{self.base_prompt}\n"
            f"Situation: \"{last_message}\"\n"
            f"Lesson plan: {state['lesson_plan']}\n"
            f"Students:\n{student_list}"
            "Which of these students is most likely to respond suitably? Return only the index number."
        )
        answer = llm.invoke(prompt).content.strip()
        try:
            return int(answer)
        except ValueError:
            # fallback to random if invalid index
            return random.randrange(len(students))

# Node functions
def init_state(state: SimulationState) -> dict:
    state['rounds'] = 3
    state['lesson_plan'] = (
        "The lesson plan is for all students to actively participate in a basic explanation of fractions "
        "and their representation in simple problems. Neither teacher nor students should exceed 20 words in "
        "their responses, and most responses should be under 20 words. Students must use short phrases typical "
        "of classroom language. Participation can be collaborative or individual. Some students may joke, interrupt, "
        "tease each other or the teacher, or try to derail the conversation. The teacher, in a firm and serious tone, "
        "applies negative punishment by removing privileges (e.g., deducting recess minutes or exam points) whenever "
        "there is disrespect or disruption. Nonsense or long digressions are not allowed. All exchanges should maintain "
        "a believable classroom tone: brief sentences, youthful vocabulary from students, and terse responses from the teacher. "
        "The goal is for the group to understand fractions, while the teacher maintains order against disruptive behavior. "
        "It should not feel theatrical: do not describe physical gestures."
    )
    state['current_round'] = 0
    state['last_message'] = (
        "Welcome to Math class! Today's goal is to learn how to use fractions and why they're useful. "
        "I hope you enjoy it—I have prepared many practical exercises."
    )
    return state

def select_student(state: SimulationState) -> dict:
    moderator: ModeratorAgent = state['moderator']
    idx = moderator.select(state['students'], state['last_message'], state)
    state['selected_index'] = idx
    return state

def student_act(state: SimulationState) -> dict:
    idx = state['selected_index']
    student = state['students'][idx]
    response = student.act(state['last_message'], state)
    print(f"{student.name}: {response}")
    state['last_message'] = f"{student.name}: {response}"
    return state

def teacher_respond(state: SimulationState) -> dict:
    idx = state['selected_index']
    student = state['students'][idx]
    teacher = state['teacher']
    student_msg = state['last_message']
    response = teacher.respond(student_msg, student, state)
    print(f"{teacher.name}: {response}")
    state['last_message'] = f"{teacher.name}: {response}"
    return state

def increment_round(state: SimulationState) -> dict:
    state['current_round'] += 1
    if state['current_round'] >= state['rounds']:
        print("Simulation ended.")
    else:
        print(f"--- Round {state['current_round']} ---")
    return state

# Instantiate agents based on reports and articles
students = [
    StudentAgent(
        name="Ana",
        age=14,
        profile="reserved, perfectionist, often corrects classmates when they err",
        base_prompt=(
            "You are Ana, a 14-year-old student who is reserved and perfectionist. "
            "You take your studies very seriously and quickly spot errors in others, "
            "which you point out matter-of-factly. You dislike class interruptions or "
            "slow pace. When you get an explanation, you analyze its precision and "
            "correct inaccuracies in a serious tone. You participate only if you have "
            "something important to say. If the teacher removes a privilege for your "
            "behavior, you perceive it as unfair and point it out firmly but respectfully."
        )
    ),
    StudentAgent(
        name="Luis",
        age=15,
        profile="chatty, charismatic, easily distracted, always tries to make jokes",
        base_prompt=(
            "You are Luis, a 15-year-old student who is charismatic and talkative. "
            "Though interested in the lesson, you can't help cracking jokes or making funny "
            "comments to entertain classmates. You struggle to stay focused and are often "
            "distracted by things outside class. If the teacher punishes by deducting recess "
            "minutes, you pretend not to care, but then you try to lift the group's mood with another joke."
        )
    ),
    StudentAgent(
        name="Claudia",
        age=14,
        profile="impulsive, creative, constantly interrupts with off-topic questions",
        base_prompt=(
            "You are Claudia, a 14-year-old student who is very creative and impulsive. "
            "You have many ideas and questions, but they are not always related to class. "
            "You tend to interrupt with unexpected or provocative queries out of genuine curiosity. "
            "If the teacher punishes you, you may react with irony or feigned indifference, "
            "and sometimes use the situation to ask an even more irreverent question."
        )
    ),
    StudentAgent(
        name="Javi",
        age=15,
        profile="rebellious, outspoken, questions everything the teacher says",
        base_prompt=(
            "You are Javi, a 15-year-old student who questions everything the teacher says. "
            "You don't like following rules if you don't understand their purpose. "
            "You speak out of turn and use informal, sarcastic language. You enjoy provoking reactions. "
            "If the teacher punishes by removing privileges, you take it as a challenge and make biting remarks. "
            "Sometimes you get other students to join your interruptions."
        )
    ),
    StudentAgent(
        name="Nerea",
        age=13,
        profile="restless, funny, loves being the center of attention",
        base_prompt=(
            "You are Nerea, a 13-year-old student who is restless and craves attention. "
            "You speak even without being called on, often saying things unrelated to class. "
            "You love entertaining the group and being the center of attention. "
            "If the teacher punishes you, you pretend not to care but quickly crack a joke to shift focus."
        )
    )
]

teacher = TeacherAgent(
    name="Adrián Herrero",
    subject="Mathematics",
    methodology=(
        "Negative punishment: The teacher removes something pleasant or desired by the student when they misbehave. "
        "For example, the teacher may deduct recess minutes, exam points, or a small perk immediately after an infraction, "
        "thus removing a positive stimulus to discourage undesirable behavior."
    ),
    base_prompt=(
        "You are Professor Adrián Herrero, a veteran Mathematics teacher with decades of experience. "
        "You are empathetic and care about your students but are firm and strict when necessary. "
        "You have noticed a decline in respect and discipline in classrooms, leading you to adopt negative punishment. "
        "You speak in a calm yet authoritative tone, emphasizing effort and mutual respect. "
        "When someone interrupts or acts disrespectfully, you immediately remove privileges to set clear boundaries. "
        "You are deeply bothered by students not listening or treating class like a game. "
        "Your goal is for them to learn but you will not tolerate behavior that disrupts the group."
    )
)

moderator = ModeratorAgent(
    base_prompt=(
        "You are the Classroom Moderator. Your job is to decide which student should respond each round, "
        "based on the teacher's last message, the lesson plan, and the students' profiles."
    )
)

# Build the workflow
builder = StateGraph(SimulationState)
builder.add_node('init_state', init_state)
builder.add_node('select_student', select_student)
builder.add_node('student_act', student_act)
builder.add_node('teacher_respond', teacher_respond)
builder.add_node('increment_round', increment_round)

# Define edges
builder.add_edge(START, 'init_state')
builder.add_edge('init_state', 'select_student')
builder.add_edge('select_student', 'student_act')
builder.add_edge('student_act', 'teacher_respond')
builder.add_edge('teacher_respond', 'increment_round')
builder.add_conditional_edges(
    'increment_round',
    lambda state: END if state['current_round'] >= state['rounds'] else 'select_student'
)

workflow = builder.compile()

# Run simulation
final_state = workflow.invoke(
    {'students': students, 'teacher': teacher, 'moderator': moderator},
    config={"recursion_limit": 100}
)

print("Simulation completed.")
