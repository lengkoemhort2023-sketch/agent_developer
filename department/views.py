from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Department
from .serializers import (
    DepartmentSerializer, 
    DepartmentCreateSerializer, 
    DepartmentUpdateSerializer,
    DepartmentListSerializer
)
from base.utils import success_response, error_response, handle_exception
import logging

logger = logging.getLogger(__name__)

@api_view(['GET'])
@handle_exception
def list_departments(request):
    """List all departments using serializer"""
    departments = Department.objects.all()
    serializer = DepartmentListSerializer(departments, many=True)
    return success_response(message="Departments retrieved successfully", data=serializer.data)

@api_view(['POST'])
@handle_exception
def create_department(request):
    """Create a new department using serializer"""
    serializer = DepartmentCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return error_response("Invalid data provided", serializer.errors, status.HTTP_400_BAD_REQUEST)
    
    department = serializer.save()
    response_serializer = DepartmentSerializer(department)
    return success_response("Department created successfully", response_serializer.data, status.HTTP_201_CREATED)

@api_view(['GET'])
@handle_exception
def get_department(request, dept_id):
    """Get a specific department using serializer"""
    department = get_object_or_404(Department, id=dept_id)
    serializer = DepartmentSerializer(department)
    return success_response("Department retrieved successfully", serializer.data)

@api_view(['PUT'])
@handle_exception
def update_department(request, dept_id):
    """Update a department using serializer"""
    department = get_object_or_404(Department, id=dept_id)
    serializer = DepartmentUpdateSerializer(department, data=request.data, partial=True)
    if not serializer.is_valid():
        return error_response("Invalid data provided", serializer.errors, status.HTTP_400_BAD_REQUEST)
    
    department = serializer.save()
    response_serializer = DepartmentSerializer(department)
    return success_response("Department updated successfully", response_serializer.data)

@api_view(['DELETE'])
@handle_exception
def delete_department(request, dept_id):
    """Delete a department"""
    department = get_object_or_404(Department, id=dept_id)
    department.delete()
    return success_response("Department deleted successfully")







