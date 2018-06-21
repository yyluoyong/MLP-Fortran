module mod_SinXYCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
implicit none    

!------------------------------
! �����ࣺMNIST���ݼ��������� |
!------------------------------
type, extends(BaseCalculationCase), public :: SinXYCase
    !* �̳���BaseCalculationCase��ʵ����ӿ�
    
    !* �Ƿ��ʼ���ڴ�ռ�
    logical, private :: is_allocate_done = .false.
    
    !* ѵ������������
    integer, public :: count_train_sample = 201
    
    !* ���Լ���������
    integer, public :: count_test_sample = 100
    
    !* ����������������: 28 ��28 = 784
    integer, public :: sample_point_X = 2
    integer, public :: sample_point_y = 1
    
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

    procedure, private :: init_train_data   => m_init_train_data
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    final :: SinXYCase_clean_space
    
end type SinXYCase
!===================

    !-------------------------
    private :: m_main
    private :: m_init_train_data
    private :: m_allocate_memory
    private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ������
    subroutine m_main( this )
    implicit none
        class(SinXYCase), intent(inout) :: this
    
        real(kind=PRECISION), dimension(:,:), allocatable :: y
    
        call this % allocate_memory()
        
        call this % init_train_data()
        
        allocate( y, SOURCE = this % y_train)
        
        call this % my_NNTrain % init('Sin_X_Y_Case', this % X_train, &
            this % y_train)
        
        call this % my_NNTrain % train(this % X_train, &
            this % y_train, y)
        
        return
    end subroutine m_main
    !====
    
    !* ��ʼ��ѵ������
    subroutine m_init_train_data( this )
    implicit none
        class(SinXYCase), intent(inout) :: this
        
        real(PRECISION) :: PI, dx
        integer :: i
        
        PI = 4 * ATAN(1.0)
        
        associate ( &
            count_train_sample => this % count_train_sample, &
            X_train            => this % X_train,            &
            y_train            => this % y_train             &
        )
        
        dx = 2.0 / (count_train_sample - 1)
        do i=1, count_train_sample
            X_train(1, i) = -1 + (i-1) * dx
            X_train(2, i) = -1 + (i-1) * dx
            y_train(1, i) = SIN(X_train(1, i) * X_train(2, i))
        end do
        
        y_train = 0.5 * (y_train + 1)
        
        end associate
    
        return
    end subroutine m_init_train_data
    !====

    !* �����ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(SinXYCase), intent(inout) :: this
        
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
        class(SinXYCase), intent(inout)  :: this	
        
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
    subroutine SinXYCase_clean_space( this )
    implicit none
        type(SinXYCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("SinXYCase: SUBROUTINE clean_space.")
        
        return
    end subroutine SinXYCase_clean_space
    !====
    
end module