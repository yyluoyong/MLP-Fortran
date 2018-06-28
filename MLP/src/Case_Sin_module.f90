module mod_SinCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
use mod_Tools
use mod_MeanSquareError
implicit none    

!-------------------
! �����ࣺ���Һ��� |
!-------------------
type, extends(BaseCalculationCase), public :: SinCase
    !* �̳���BaseCalculationCase��ʵ����ӿ�
    
    !* �Ƿ��ʼ���ڴ�ռ�
    logical, private :: is_allocate_done = .false.
    
    !* ѵ������������
    integer, public :: count_train_sample = 201
    
    !* ���Լ���������
    integer, public :: count_test_sample = 100
    
    !* ����������������: 28 ��28 = 784
    integer, public :: sample_point_X = 1
    integer, public :: sample_point_y = 1
    
    !* ѵ�����ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* ѵ�����ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    !* Ԥ����
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_pre
    
    !* �������ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* �������ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    
    type(NNTrain), pointer :: my_NNTrain
	type(MeanSquareError), pointer :: mse_function
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: main => m_main

    procedure, private :: init_train_data   => m_init_train_data
    procedure, private :: output_data       => m_output_data
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    final :: SinCase_clean_space
    
end type SinCase
!===================

    !-------------------------
    private :: m_main
    private :: m_init_train_data
    private :: m_output_data
    private :: m_allocate_memory
    private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ������
    subroutine m_main( this )
    implicit none
        class(SinCase), intent(inout) :: this

        call this % allocate_memory()
        
        call this % init_train_data()
		
		associate (                            &
            X_train     => this % X_train,     &
            y_train     => this % y_train,     &
            y_train_pre => this % y_train_pre  &              
        )
        
        call this % my_NNTrain % init('SinCase', &
            this % sample_point_X, this % sample_point_y)   
        
        call this % my_NNTrain % &
            set_weight_threshold_init_methods_name('xavier') 

		call this % my_NNTrain % set_loss_function(this % mse_function)
        
        call this % my_NNTrain % train(X_train, y_train, y_train_pre)
        
        call this % output_data()
        
        end associate
        
        return
    end subroutine m_main
    !====
    
    !* ��ʼ��ѵ������
    subroutine m_init_train_data( this )
    implicit none
        class(SinCase), intent(inout) :: this
        
        real(PRECISION) :: PI, dx
        integer :: i
        
        PI = 4 * ATAN(1.0)
        
        associate (                                          &
            count_train_sample => this % count_train_sample, &
            X_train            => this % X_train,            &
            y_train            => this % y_train             &
        )
        
        dx = 1.0 / (count_train_sample - 1)
        do i=1, count_train_sample
            X_train(1, i) = (i-1) * dx
            y_train(1, i) = SIN(2 * PI * X_train(1, i))
        end do
        
        y_train = 0.25 * (y_train + 1)
        
        end associate
    
        return
    end subroutine m_init_train_data
    !====

        !* �����������
    subroutine m_output_data( this )
    implicit none
        class(SinCase), intent(inout) :: this
    
        real(kind=PRECISION), dimension(:), allocatable :: Y_err
        integer :: i, j
            
        associate (                                          &
            count_train_sample => this % count_train_sample, &
            X_train            => this % X_train,            &
            y_train            => this % y_train,            &
            y_train_pre        => this % y_train_pre         &
        )
        
        allocate( Y_err(count_train_sample) )
      
        
        Y_err = abs(y_train(1,:) - y_train_pre(1,:))
        
        call output_tecplot_line(                        &                       
            './Output/SinCase/result.plt',               &
            'X', X_train(1,:), 'Sin_True', y_train(1,:), &
            'Sin_Pre', y_train_pre(1,:), 'Sin_Err', Y_err      &
        )
            
        deallocate(Y_err)
            
        end associate
               
        return
    end subroutine m_output_data
    !====
    
    
    !* �����ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(SinCase), intent(inout) :: this
        
        associate (                                              &
                sample_point_X     => this % sample_point_X,     &
                sample_point_y     => this % sample_point_y,     &
                count_train_sample => this % count_train_sample, &
                count_test_sample  => this % count_test_sample   &              
        )
        
        allocate( this % X_train(sample_point_X, count_train_sample) )        
        allocate( this % y_train(sample_point_y, count_train_sample) )
        allocate( this % y_train_pre(sample_point_y, count_train_sample) )
        
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
        class(SinCase), intent(inout)  :: this	
        
        deallocate( this % X_train )        
        deallocate( this % y_train )
        deallocate( this % y_train_pre )
        
        deallocate( this % X_test )
        deallocate( this % y_test )    
        
        deallocate( this % my_NNTrain )
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* ���������������ڴ�ռ�
    subroutine SinCase_clean_space( this )
    implicit none
        type(SinCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("SinCase: SUBROUTINE clean_space.")
        
        return
    end subroutine SinCase_clean_space
    !====
    
end module