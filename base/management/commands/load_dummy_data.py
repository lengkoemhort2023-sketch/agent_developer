from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from department.models import Department
from docs_type.models import DocumentType
from document.models import Document
from chat.models import ChatSession, ChatMessage, ChatInput
import uuid
from datetime import datetime, timedelta
import random

User = get_user_model()

class Command(BaseCommand):
    help = 'Load dummy data for testing and development'

    def add_arguments(self, parser):
        parser.add_argument('--users', type=int, default=5, help='Number of users to create')
        parser.add_argument('--departments', type=int, default=3, help='Number of departments to create')
        parser.add_argument('--documents', type=int, default=10, help='Number of documents to create')
        parser.add_argument('--chat-sessions', type=int, default=8, help='Number of chat sessions to create')

    def handle(self, *args, **options):
        self.stdout.write('Starting to load dummy data...')
        self.create_regular_dummy_data(options)

    def create_regular_dummy_data(self, options):
        users = self.create_users(options['users'])
        departments = self.create_departments(options['departments'])
        doc_types = self.create_document_types()
        documents = self.create_documents(options['documents'], departments, doc_types, users)
        chat_sessions = self.create_chat_sessions(options['chat_sessions'], users)
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created:\n'
                f'- {len(users)} users\n'
                f'- {len(departments)} departments\n'
                f'- {len(doc_types)} document types\n'
                f'- {len(documents)} documents\n'
                f'- {len(chat_sessions)} chat sessions'
            )
        )

    def create_users(self, count):
        users = []
        first_names = ['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Tom', 'Emma', 'Alex', 'Maria']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        for i in range(count):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            username = f"{first_name.lower()}{last_name.lower()}{i+1}"
            email = f"{username}@example.com"
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    'first_name': first_name,
                    'last_name': last_name,
                    'employee_id': f"EMP{i+1:03d}",
                    'is_active': True,
                    'email': email,  # Add this line
                }
            )
            users.append(user)
            if created:
                self.stdout.write(f'Created user: {user.first_name} {user.last_name}')
        return users

    def create_departments(self, count):
        departments = []
        dept_names = [
            'Information Technology',
            'Human Resources', 
            'Finance',
            'Marketing',
            'Operations',
            'Sales',
            'Research & Development',
            'Customer Support'
        ]
        for i in range(count):
            name = dept_names[i] if i < len(dept_names) else f"Department {i+1}"
            if i == 0:
                dept, created = Department.objects.get_or_create(
                    id='28249356-b6d4-4311-8940-98c7c5095f70',
                    defaults={'name': name}
                )
            else:
                dept, created = Department.objects.get_or_create(name=name)
            departments.append(dept)
            if created:
                self.stdout.write(f'Created department: {dept.name}')
        return departments

    def create_document_types(self):
        doc_types = []
        type_names = [
            'Policy Document',
            'Procedure Manual',
            'Training Material',
            'Report',
            'Contract',
            'Invoice',
            'Certificate',
            'Guideline'
        ]
        for i, name in enumerate(type_names):
            if i == 0:
                doc_type, created = DocumentType.objects.get_or_create(
                    id='8bd36abb-96dc-45d5-ad35-9dfe6741fc60',
                    defaults={'presentation': name}
                )
            else:
                doc_type, created = DocumentType.objects.get_or_create(presentation=name)
            doc_types.append(doc_type)
            if created:
                self.stdout.write(f'Created document type: {doc_type.presentation}')
        return doc_types

    def create_documents(self, count, departments, doc_types, users):
        documents = []
        titles = [
            'Employee Handbook',
            'IT Security Policy',
            'Financial Report Q1',
            'Marketing Strategy',
            'Training Manual',
            'Project Proposal',
            'Client Contract',
            'Quality Assurance Guide',
            'Safety Procedures',
            'Annual Report'
        ]
        for i in range(count):
            title = titles[i] if i < len(titles) else f"Document {i+1}"
            version = random.randint(1, 3)
            doc = Document.objects.create(
                title=title,
                description=f"This is a sample {title.lower()}",
                published_date=datetime.now().date() - timedelta(days=random.randint(1, 365)),
                department=random.choice(departments),
                type=random.choice(doc_types),
                version=version,
                original_filename=f"{title.lower().replace(' ', '_')}_v{version}.pdf",
                stored_filename=f"doc_{uuid.uuid4().hex[:8]}.pdf",
                file_path=f"documents/{title.lower().replace(' ', '_')}_v{version}.pdf",
                file_size=random.randint(100000, 5000000),
                file_format='pdf',
                is_active=True
            )
            documents.append(doc)
            self.stdout.write(f'Created document: {doc.title} (v{doc.version})')
        return documents

    def create_chat_sessions(self, count, users):
        chat_sessions = []
        questions = [
            "Hello, how are you?",
            "Can you help me with a document?",
            "What's the weather like?",
            "How do I submit a report?",
            "Where can I find the employee handbook?",
            "What are the company policies?",
            "How do I request time off?",
            "Can you explain the benefits package?",
            "What's the process for expense reports?",
            "How do I access the training materials?"
        ]
        answers = [
            "Hello! I'm doing well, thank you for asking. How can I help you today?",
            "Of course! I'd be happy to help you with any document-related questions.",
            "I don't have access to real-time weather information, but I can help you with other queries.",
            "You can submit reports through the online portal or by contacting your supervisor.",
            "The employee handbook is available in the documents section of our system.",
            "Company policies are outlined in the policy documents section. Would you like me to help you find a specific policy?",
            "You can request time off through the HR portal or by contacting your manager directly.",
            "The benefits package includes health insurance, retirement plans, and paid time off. I can provide more details if needed.",
            "Expense reports can be submitted through the finance portal with proper receipts and approvals.",
            "Training materials are available in the learning management system under your department's section."
        ]
        for i in range(count):
            user = random.choice(users)
            session = ChatSession.objects.create(
                user=user
            )
            chat_sessions.append(session)
            message_count = random.randint(2, 5)
            for j in range(message_count):
                question = random.choice(questions)
                answer = random.choice(answers)
                message = ChatMessage.objects.create(
                    session=session,
                    question=question,
                    answer=answer
                )
                if random.choice([True, False]):
                    ChatInput.objects.create(
                        message=message,
                        user=user,
                        input_type=random.choice(['text', 'voice']),
                        content=question,
                        voice_duration=random.uniform(1.0, 5.0) if random.choice([True, False]) else None
                    )
            self.stdout.write(f'Created chat session for {user.first_name} with {message_count} messages')
        return chat_sessions







