module mod_MoonCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
implicit none    

!-----------------------------
! �����ࣺMoon���ݼ��������� |
!-----------------------------
type, extends(BaseCalculationCase), public :: MoonCase
    !* �̳���BaseCalculationCase��ʵ����ӿ�
    
    character(len=180), private :: train_data_file = &
        './Data/MoonCase/moon_data.txt'

    !* �Ƿ��ʼ���ڴ�ռ�
    logical, private :: is_allocate_done = .false.
    
    !* ѵ������������
    integer, public :: count_train_sample = 2000
    
    !* ���Լ���������
    integer, public :: count_test_sample = 1000
    
    !* ����������������
    integer, public :: sample_point_X = 2
    integer, public :: sample_point_y = 2
    
    !* ѵ�����ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* ѵ�����ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    
    !* �������ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* �������ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    
    type(NNTrain), pointer :: my_NNTrain
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: main => m_main

    procedure, private :: load_Moon_data => m_load_Moon_data
    procedure, private :: normalization  => m_normalization
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    final :: MoonCase_clean_space
    
end type MoonCase
!===================

    !-------------------------
    private :: m_main
    private :: m_load_Moon_data
    private :: m_normalization
    private :: m_allocate_memory
    private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ������
    subroutine m_main( this )
    implicit none
        class(MoonCase), intent(inout) :: this
    
        real(PRECISION), dimension(:,:), allocatable :: y
    
        call this % allocate_memory()
        
        allocate( y, SOURCE = this % y_train)
        
        call this % load_Moon_data()
        
        call this % normalization(this % X_train)
        
        call this % my_NNTrain % train('MoonCase', this % X_train, &
            this % y_train, y)
        
        return
    end subroutine m_main
    !====
    
    !* �����ݹ�һ��
    subroutine m_normalization( this, X )
    implicit none
        class(MoonCase), intent(inout) :: this
        real(PRECISION), dimension(:,:), intent(inout) :: X
    
        real(PRECISION) :: max_x, min_x
        integer :: X_shape(2), j
        
        X_shape = SHAPE(X)
        
        min_x = MINVAL(X(1,:))
        max_x = MAXVAL(X(1,:))
        
        do j=1, X_shape(2)
            X(1, j) = (X(1, j) - min_x) / (max_x - min_x)
        end do

        min_x = MINVAL(X(2,:))
        max_x = MAXVAL(X(2,:))
        
        do j=1, X_shape(2)
            X(2, j) = (X(2, j) - min_x) / (max_x - min_x)
        end do
        
        
        return
    end subroutine m_normalization
    !====    
    
    !* ��ȡMoon����
    subroutine m_load_Moon_data( this )
    implicit none
        class(MoonCase), intent(inout) :: this
    
        integer :: j
        real(PRECISION) :: label
        
        associate ( &
            X_train            => this % X_train,            &
            y_train            => this % y_train,            &
            X_test             => this % X_test,             &
            y_test             => this % y_test,             &
            count_train_sample => this % count_train_sample, &
            count_test_sample  => this % count_test_sample   &              
        )
        
        open(UNIT=30, FILE=this % train_data_file, &
            FORM='formatted', STATUS='old')
        
        y_train = 0
        do j=1, count_train_sample
            read(30, *) X_train(1, j), X_train(2, j), label
            if (label > 0) then 
                y_train(1, j) = 1
            else
                y_train(2, j) = 1
            end if
        end do
        
        y_test = 0
        do j=1, count_test_sample
            read(30, *) X_test(1, j), X_test(2, j), label
            if (label > 0) then 
                y_test(1, j) = 1
            else
                y_test(2, j) = 1
            end if
        end do
            
        close(30)
        
        end associate
        
        return
    end subroutine m_load_Moon_data
    !====
    
    !* �����ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(MoonCase), intent(inout) :: this
        
        associate ( &
            sample_point_X     => this % sample_point_X,     &
            sample_point_y     => this % sample_point_y,     &
            count_train_sample => this % count_train_sample, &
            count_test_sample  => this % count_test_sample   &              
        )
        
        allocate( this % X_train(sample_point_X, count_train_sample) )        
        allocate( this % y_train(sample_point_y, count_train_sample) )
        
        allocate( this % X_test(sample_point_X, count_test_sample) )
        allocate( this % y_test(sample_point_y, count_test_sample) )       
        
        end associate
        
        allocate( this % my_NNTrain )
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(MoonCase), intent(inout)  :: this	
        
        deallocate( this % X_train )        
        deallocate( this % y_train )
        
        deallocate( this % X_test )
        deallocate( this % y_test )    
        
        deallocate( this % my_NNTrain )
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* ���������������ڴ�ռ�
    subroutine MoonCase_clean_space( this )
    implicit none
        type(MoonCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("MoonCase: SUBROUTINE clean_space.")
        
        return
    end subroutine MoonCase_clean_space
    !====
    
end module