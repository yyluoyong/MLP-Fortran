module mod_MoonCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
use mod_CrossEntropy
use mod_SimpleBatchGenerator
implicit none    

!-----------------------------
! 工作类：Moon数据集计算算例 |
!-----------------------------
type, extends(BaseCalculationCase), public :: MoonCase
    !* 继承自BaseCalculationCase并实现其接口
    
    character(len=180), private :: train_data_file = &
        !'./Data/MoonCase/moon_data.txt'
        './Data/MoonCase/random_double_moon_data.dat'

    !* 是否初始化内存空间
    logical, private :: is_allocate_done = .false.
    
    !* 每组样本的数量
    integer, public :: batch_size = 20
    
    !* 训练集样本数量
    integer, public :: count_train_sample = 1000
    
    !* 测试集样本数量
    integer, public :: count_test_sample = 2000
    
    !* 单个样本的数据量
    integer, public :: sample_point_X = 2
    integer, public :: sample_point_y = 2
    
    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_batch
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch
    !* 训练数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch_pre
   
    
    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    !* 训练数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_pre
    
    !* 测试数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* 测试数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    !* 测试数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test_pre
    
    type(NNTrain), pointer :: my_NNTrain
    
    type(CrossEntropyWithSoftmax), pointer :: cross_entropy_function
    
    type(SimpleBatchGenerator), pointer :: batch_generator
    
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

    !* 主函数
    subroutine m_main( this )
    implicit none
        class(MoonCase), intent(inout) :: this
        
        integer :: train_count = 1
        integer :: round_step
    
        call this % allocate_memory()
        
        call this % load_Moon_data()
        
        associate (                            &
            X_batch     => this % X_batch,     &
            y_batch     => this % y_batch,     &
            y_batch_pre => this % y_batch_pre, &
            X_train     => this % X_train,     &
            y_train     => this % y_train,     &
            y_train_pre => this % y_train_pre, &
            X_test      => this % X_test,      &
            y_test      => this % y_test,      &
            y_test_pre  => this % y_test_pre   & 
        )
        
        call this % normalization(X_train)
        
        call this % my_NNTrain % init('MoonCase', &
            this % sample_point_X, this % sample_point_y)
        
        call this % my_NNTrain % set_train_type('classification')
        
        call this % my_NNTrain % &
            set_weight_threshold_init_methods_name('xavier')
            
        call this % my_NNTrain % set_loss_function(this % cross_entropy_function)
        
        do round_step=1, train_count
            call this % batch_generator % get_next_batch( &
                X_train, y_train, X_batch, y_batch )
            
            call this % my_NNTrain % train(X_batch, &
                y_batch, y_batch_pre)           
        end do
        
        !call this % my_NNTrain % init('MoonCase', X_train, y_train)
        !
        !call this % my_NNTrain % set_train_type('classification')
        !
        !call this % my_NNTrain % &
        !    set_weight_threshold_init_methods_name('xavier')
        !    
        !call this % my_NNTrain % set_loss_function(this % cross_entropy_function)
        !    
        !call this % my_NNTrain % train(X_train, &
        !    y_train, y_train_pre)
        !
        call this % normalization(X_test)
            
        call this % my_NNTrain % sim(X_test, &
            y_test, y_test_pre)
        
        end associate
            
        return
    end subroutine m_main
    !====
    
    !* 将数据归一化
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
            X(1, j) = 2 * X(1, j) - 1
        end do

        min_x = MINVAL(X(2,:))
        max_x = MAXVAL(X(2,:))
        
        do j=1, X_shape(2)
            X(2, j) = (X(2, j) - min_x) / (max_x - min_x)
            X(2, j) = 2 * X(2, j) - 1
        end do
        
        
        return
    end subroutine m_normalization
    !====    
    
    !* 读取Moon数据
    subroutine m_load_Moon_data( this )
    implicit none
        class(MoonCase), intent(inout) :: this
    
        integer :: j
        real(PRECISION) :: label
        
        associate (                                          &
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
    
    !* 申请内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(MoonCase), intent(inout) :: this
        
        associate (                                          &
            sample_point_X     => this % sample_point_X,     &
            sample_point_y     => this % sample_point_y,     &
            count_train_sample => this % count_train_sample, &
            count_test_sample  => this % count_test_sample,  &
            batch_size         => this % batch_size          &
        )
        
        allocate( this % X_train(sample_point_X, count_train_sample) )        
        allocate( this % y_train(sample_point_y, count_train_sample) )
        allocate( this % y_train_pre(sample_point_y, count_train_sample) )
        
        allocate( this % X_test(sample_point_X, count_test_sample) )
        allocate( this % y_test(sample_point_y, count_test_sample) )
        allocate( this % y_test_pre(sample_point_y, count_test_sample) ) 
        
        allocate( this % X_batch(sample_point_X, batch_size) )        
        allocate( this % y_batch(sample_point_y, batch_size) )
        allocate( this % y_batch_pre(sample_point_y, batch_size) )
      
        end associate
        
        allocate( this % my_NNTrain )
        
        allocate( this % cross_entropy_function )
        
        allocate( this % batch_generator )
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(MoonCase), intent(inout)  :: this	
        
        deallocate( this % X_train )        
        deallocate( this % y_train )
        deallocate( this % y_train_pre )
        
        deallocate( this % X_test )
        deallocate( this % y_test ) 
        deallocate( this % y_test_pre )
        
        deallocate( this % X_batch )        
        deallocate( this % y_batch )
        deallocate( this % y_batch_pre )
        
        deallocate( this % my_NNTrain )
        deallocate( this % cross_entropy_function )
        deallocate( this % batch_generator )
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* 析构函数，清理内存空间
    subroutine MoonCase_clean_space( this )
    implicit none
        type(MoonCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("MoonCase: SUBROUTINE clean_space.")
        
        return
    end subroutine MoonCase_clean_space
    !====
    
end module