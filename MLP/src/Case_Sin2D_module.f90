module mod_Sin2DCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
use mod_Tools
implicit none    

!--------------------------
! 工作类：测试一下二维数据 |
!--------------------------
type, extends(BaseCalculationCase), public :: Sin2DCase
    !* 继承自BaseCalculationCase并实现其接口
    
    !* 是否初始化内存空间
    logical, private :: is_allocate_done = .false.
    
    !* 训练集样本数量
    integer, public :: i_count = 101
    integer, public :: j_count = 101
    integer, public :: count_train_sample
    
    !* 测试集样本数量
    integer, public :: count_test_sample = 100
    
    !* 单个样本的数据量
    integer, public :: sample_point_X = 2
    integer, public :: sample_point_y = 2
    
    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    !* 接收神经网络的输出结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_nn
    
    !* 测试数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* 测试数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    
    type(NNTrain), pointer :: my_NNTrain
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: main => m_main

    procedure, private :: init_train_data   => m_init_train_data
    procedure, private :: output_data       => m_output_data
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    final :: Sin2DCase_clean_space
    
end type Sin2DCase
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

    !* 主函数
    subroutine m_main( this )
    implicit none
        class(Sin2DCase), intent(inout) :: this
    
        this % count_train_sample = this % i_count * this % j_count
        
        call this % allocate_memory()
        
        call this % init_train_data()        
        
        call this % my_NNTrain % init('2DSinCase', this % X_train, &
            this % y_train)
        
        call this % my_NNTrain % train(this % X_train, &
            this % y_train, this % y_train_nn)
            
        call this % output_data()
        
        return
    end subroutine m_main
    !====
    
    
    !* 输出计算数据
    subroutine m_output_data( this )
    implicit none
        class(Sin2DCase), intent(inout) :: this
    
        real(kind=PRECISION), dimension(:,:), allocatable :: I_grid, J_grid
        real(kind=PRECISION), dimension(:,:,:), allocatable :: Y_t, Y_nn, Y_err
        integer :: i, j
            
        associate (                                          &
            i_count            => this % i_count,            &
            j_count            => this % j_count,            &
            count_train_sample => this % count_train_sample, &
            X_train            => this % X_train,            &
            y_train            => this % y_train,            &
            y_train_nn         => this % y_train_nn          &
        )
        
        allocate( I_grid(i_count, j_count) )
        allocate( J_grid(i_count, j_count) )
        allocate( Y_t(2, i_count, j_count) )
        allocate( Y_nn(2, i_count, j_count) )
        allocate( Y_err(2, i_count, j_count) )
        
        I_grid = RESHAPE(X_train(1,:), (/ i_count, j_count /))
        J_grid = RESHAPE(X_train(2,:), (/ i_count, j_count /))
        
        Y_t(1, :, :) = RESHAPE(y_train(1,:), (/ i_count, j_count /)) 
        Y_t(2, :, :) = RESHAPE(y_train(2,:), (/ i_count, j_count /)) 
        
        Y_nn(1, :, :) = RESHAPE(y_train_nn(1,:), (/ i_count, j_count /)) 
        Y_nn(2, :, :) = RESHAPE(y_train_nn(2,:), (/ i_count, j_count /)) 
        
        Y_err = abs(Y_t - Y_nn)
        
        call output_tecplot_2D(              &                       
            './Output/2DSinCase/result.plt', &
            'X', I_grid, 'Y', J_grid,        &
            'Sin_True', Y_t(1, :, :), 'Cos_True', Y_t(2, :, :),   &
            'Sin_Pre',  Y_nn(1, :, :), 'Cos_Pre', Y_nn(2, :, :),  &
            'Sin_Err',  Y_err(1, :, :), 'Cos_Err', Y_err(2, :, :) &
        )
        
        deallocate( I_grid )
        deallocate( J_grid )
        deallocate( Y_t )
        deallocate( Y_nn )
        deallocate( Y_err )
        
        end associate
        
        return
    end subroutine m_output_data
    !====
    
    
    !* 初始化训练数据
    subroutine m_init_train_data( this )
    implicit none
        class(Sin2DCase), intent(inout) :: this
        
        real(PRECISION) :: PI, dx, dy
        integer :: i, j, index
        
        PI = 4 * ATAN(1.0)
        
        associate (                                          &
            i_count            => this % i_count,            &
            j_count            => this % j_count,            &
            count_train_sample => this % count_train_sample, &
            X_train            => this % X_train,            &
            y_train            => this % y_train             &
        )
        
        !* 输入：X × Y ∈ [0, 1] × [0, 1]
        !* 输出：sin[2π(X+Y)] × cos[2π(X+Y)]
        dx = 1.0 / (i_count - 1)
        dy = 1.0 / (j_count - 1)
        do j=1, j_count
            do i=1, i_count
                index = i_count * (j - 1) + i
                
                X_train(1, index) = (i-1) * dx
                X_train(2, index) = (j-1) * dy
                
                y_train(1, index) = SIN(2*PI*( X_train(1, i) + X_train(2, i) ))
                y_train(2, index) = COS(2*PI*( X_train(1, i) + X_train(2, i) ))
            end do
        end do
        
        y_train = 0.25 * (y_train + 1.1)
        
        end associate
    
        return
    end subroutine m_init_train_data
    !====

    !* 申请内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(Sin2DCase), intent(inout) :: this
        
        associate ( &
                sample_point_X     => this % sample_point_X,     &
                sample_point_y     => this % sample_point_y,     &
                count_train_sample => this % count_train_sample, &
                count_test_sample  => this % count_test_sample   &              
        )
        
        allocate( this % X_train(sample_point_X, count_train_sample) )        
        allocate( this % y_train(sample_point_y, count_train_sample) )
        allocate( this % y_train_nn(sample_point_y, count_train_sample) )
        
        allocate( this % X_test(sample_point_X, count_test_sample) )
        allocate( this % y_test(sample_point_y, count_test_sample) )       
        
        end associate
        
        allocate( this % my_NNTrain )
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(Sin2DCase), intent(inout)  :: this	
        
        deallocate( this % X_train )        
        deallocate( this % y_train )
        deallocate( this % y_train_nn )
        
        deallocate( this % X_test )
        deallocate( this % y_test )    
        
        deallocate( this % my_NNTrain )
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* 析构函数，清理内存空间
    subroutine Sin2DCase_clean_space( this )
    implicit none
        type(Sin2DCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("Sin2DCase: SUBROUTINE clean_space.")
        
        return
    end subroutine Sin2DCase_clean_space
    !====
    
end module