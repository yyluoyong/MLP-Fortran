module mod_MNISTCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
use mod_CrossEntropy
use mod_SimpleBatchGenerator
use mod_ShuffleBatchGenerator
use mod_OptimizationAdam
implicit none    

!------------------------------
! 工作类：MNIST数据集计算算例 |
!------------------------------
type, extends(BaseCalculationCase), public :: MNISTCase
    !* 继承自BaseCalculationCase并实现其接口
    
    character(len=180), private :: train_image_data_file = &
        './Data/MNISTCase/train-images.fortran'
    character(len=180), private :: train_label_data_file = &
        './Data/MNISTCase/train-labels.fortran'
    character(len=180), private :: test_image_data_file = &
        './Data/MNISTCase/t10k-images.fortran'
    character(len=180), private :: test_label_data_file = &
        './Data/MNISTCase/t10k-labels.fortran'

    !* 是否初始化内存空间
    logical, private :: is_allocate_done = .false.
	
	!* 每组样本的数量
    integer, public :: batch_size = 100
    
    !* 原始数据训练集样本数量，最大是60000
	integer, public :: count_train_origin = 60000
	!* 训练集样本数量
    integer, public :: count_train = 55000
	
	!* 验证集样本数量
	integer, public :: count_validation = 5000
    
    !* 测试集样本数量，最大是10000
    integer, public :: count_test = 5000
    
    !* 单个样本的数据量: 28 ×28 = 784
    integer, public :: sample_point_X = 784
    integer, public :: sample_point_y = 10
	
	!* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_batch
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch
    !* 训练数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch_pre
    
	!* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train_origin
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_origin
	
    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    !* 训练数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_pre
    
	!* 验证数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_validate
    !* 验证数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_validate
    !* 验证数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_validate_pre
	
    !* 测试数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* 测试数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    !* 测试数据的预测结果
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test_pre
    
    type(NNTrain), pointer :: my_NNTrain
    
    type(CrossEntropyWithSoftmax), pointer :: cross_entropy_function
	
	!type(SimpleBatchGenerator), pointer :: batch_generator
    type(ShuffleBatchGenerator), pointer :: batch_generator
	
	type(OptimizationAdam), pointer :: adam_method
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: main => m_main

    procedure, private :: load_MNIST_data => m_load_MNIST_data
    procedure, private :: read_MNIST_data_from_file => m_read_MNIST_data_from_file
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    final :: MNISTCase_clean_space
    
end type MNISTCase
!===================

    !-------------------------
    private :: m_main
    private :: m_load_MNIST_data
    private :: m_read_MNIST_data_from_file
    private :: m_allocate_memory
    private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* 主函数
    subroutine m_main( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
		
		integer :: train_count = 10000
        integer :: round_step
        character(len=20) :: round_step_to_str
		integer :: train_sub_count
    
        call Log_set_file_name_prefix("MNIST")

        call this % allocate_memory()
        
        call this % load_MNIST_data()
        
        associate (                                          &
            X_train_origin     => this % X_train_origin,     &
            y_train_origin     => this % y_train_origin,     &
            X_batch            => this % X_batch,            &
            y_batch            => this % y_batch,            &
            y_batch_pre        => this % y_batch_pre,        &
            X_train            => this % X_train,            &
            y_train            => this % y_train,            &
            y_train_pre        => this % y_train_pre,        &
			X_validate         => this % X_validate,         &
            y_validate         => this % y_validate,         &
            y_validate_pre     => this % y_validate_pre,     &
            X_test             => this % X_test,             &
            y_test             => this % y_test,             &
            y_test_pre         => this % y_test_pre,         & 
			my_NNTrain         => this % my_NNTrain,         &
			count_train        => this % count_train,        &
			count_train_origin => this % count_train_origin, &
            count_test         => this % count_test,         &
			count_validate     => this % count_validation,   &
			batch_size         => this % batch_size,         &	
            sample_point_X     => this % sample_point_X,     &
            sample_point_y     => this % sample_point_y      &
        )   
        
        !----------------------------------------	        
		X_train = X_train_origin(:, 1:count_train)
		y_train = y_train_origin(:, 1:count_train)
        
        X_train = X_train / 128.0 - 1.0
		X_test  = X_test  / 128.0 - 1.0	
		
		train_sub_count = count_train - count_validate
		
		X_validate = X_train_origin(:, train_sub_count+1:count_train_origin)
		y_validate = y_train_origin(:, train_sub_count+1:count_train_origin)
		!----------------------------------------
		
		
		!----------------------------------------
		call my_NNTrain % init('MNISTCase', sample_point_X, sample_point_y)
        
        call my_NNTrain % set_train_type('classification')        
        call my_NNTrain % set_weight_threshold_init_methods_name('xavier')            
        call my_NNTrain % set_loss_function(this % cross_entropy_function)
		
		call this % adam_method % set_NN( my_NNTrain % my_NNStructure )
        !call this % adam_method % set_Adam_parameter(eps=0.01)
		call my_NNTrain % set_optimization_method( this % adam_method )
        call my_NNTrain % set_train_msg_output_step(100)
		!----------------------------------------
		
		
		do round_step=1, train_count     
            
            call this % batch_generator % get_next_batch( &
                X_train, y_train, X_batch, y_batch )
            
            call LogInfo("Batch Set: ")
            call this % my_NNTrain % sim(X_batch, y_batch, y_batch_pre)
                
            call my_NNTrain % train(X_batch, y_batch, y_batch_pre)    
            
            write(UNIT=round_step_to_str, FMT='(I15)') round_step
            call LogInfo("round_step = " // TRIM(ADJUSTL(round_step_to_str)))
            
            if (MOD(round_step, 1) == 0) then
			    call LogInfo("Validation Set: ")
                call my_NNTrain % sim(X_validate, y_validate, y_validate_pre)
			
			    call LogInfo("Test Set: ")
			    call my_NNTrain % sim(X_test, y_test, y_test_pre)
            end if

        end do
        
        !call this % my_NNTrain % train(X_train, &
        !    y_train, y_train_pre)
        !
        !call this % my_NNTrain % sim(X_train, &
        !    y_train, y_train_pre)
        !
        !call this % my_NNTrain % train(X_train, &
        !    y_train, y_train_pre)
        !
        !call this % my_NNTrain % sim(X_test, &
        !    y_test, y_test_pre)
        
            
        end associate
            
        return
    end subroutine m_main
    !====
    
    !* 读取MNIST数据
    subroutine m_load_MNIST_data( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
    
        call this % read_MNIST_data_from_file(&
            this % train_image_data_file, this % X_train_origin)
        
        call this % read_MNIST_data_from_file(&
            this % train_label_data_file, this % y_train_origin)
		
        call this % read_MNIST_data_from_file(&
            this % test_image_data_file, this % X_test)
        
        call this % read_MNIST_data_from_file(&
            this % test_label_data_file, this % y_test)
        
        return
    end subroutine m_load_MNIST_data
    !====
    
    !* 从文件中读取MNIST数据
    subroutine m_read_MNIST_data_from_file( this, file_name, data_array )
    implicit none
        class(MNISTCase), intent(inout) :: this
        character(len=*), intent(in) :: file_name
        real(PRECISION), dimension(:,:), intent(out) :: data_array

        integer(kind=4) :: magic_number, sample_count, row, column
        integer(kind=4) :: label, pixel
        integer(kind=4) , dimension(:,:), allocatable :: data_array_int4
        integer :: data_shape(2)
        integer :: i, j
    
        data_shape = SHAPE(data_array)
      
        open(UNIT=30, FILE=file_name, &
            ACCESS='stream', FORM='unformatted', STATUS='old')
        
        if (data_shape(1) == this % sample_point_y) then 
        
            allocate( data_array_int4(1, data_shape(2)) )
            
            !* 读取 label
            read(30) magic_number, sample_count
            
            if (magic_number /= 2049) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file")
                call LogErr("--> magic_number /= 2049.")
                stop
            end if
            
            read(30) (data_array_int4(1,j), j=1, data_shape(2)) 

            !* label的取值范围是：0-9
            !* 将data_array转换成one-hot形式，即：
            !* label = 0 --> [1,0,0,0,0,0,0,0,0,0]
            !* label = 1 --> [0,1,0,0,0,0,0,0,0,0]
            !* 以此类推 ... 
            !* label = 9 --> [0,0,0,0,0,0,0,0,0,1]
            data_array = 0
            do j=1, data_shape(2)
                data_array(data_array_int4(1,j)+1, j) = 1.0
            end do
            
            deallocate( data_array_int4 )
            
        else if (data_shape(1) == this % sample_point_X) then   
        
            allocate( data_array_int4(data_shape(1), data_shape(2)) )
        
            !* 读取 image
            read(30) magic_number, sample_count, row, column
            
            if (magic_number /= 2051) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file")
                call LogErr("--> magic_number /= 2051.")
                stop
            end if
            
            read(30) ((data_array_int4(i,j), i=1, data_shape(1)), j=1, data_shape(2)) 
            
            data_array = data_array_int4

            deallocate( data_array_int4 )
        else
            call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file.")
            stop
        end if

        close(30)
        
        return
    end subroutine m_read_MNIST_data_from_file
    !====

    !* 申请内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
        
        associate (                                          &
            point_X            => this % sample_point_X,     &
            point_y            => this % sample_point_y,     &
            count_train        => this % count_train,        &
			count_train_origin => this % count_train_origin, &
            count_test         => this % count_test,         &
			count_validate     => this % count_validation,   &
			batch_size         => this % batch_size          &			
        )
        
		allocate( this % X_train_origin(point_X, count_train_origin) )        
        allocate( this % y_train_origin(point_y, count_train_origin) )
		
        allocate( this % X_train(point_X, count_train) )        
        allocate( this % y_train(point_y, count_train) )
        allocate( this % y_train_pre(point_y, count_train) )
        
		allocate( this % X_validate(point_X, count_validate) )        
        allocate( this % y_validate(point_y, count_validate) )
        allocate( this % y_validate_pre(point_y, count_validate) )
		
        allocate( this % X_test(point_X, count_test) )
        allocate( this % y_test(point_y, count_test) ) 
        allocate( this % y_test_pre(point_y, count_test) ) 
		
		allocate( this % X_batch(point_X, batch_size) )        
        allocate( this % y_batch(point_y, batch_size) )
        allocate( this % y_batch_pre(point_y, batch_size) )
        
        end associate
        
        allocate( this % my_NNTrain )
        
        allocate( this % cross_entropy_function )
		
		allocate( this % batch_generator )
		
		allocate( this % adam_method )
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout)  :: this	
        
		deallocate( this % X_train_origin )        
        deallocate( this % y_train_origin )
		
        deallocate( this % X_train )        
        deallocate( this % y_train )
        deallocate( this % y_train_pre )
		
        deallocate( this % X_validate )        
        deallocate( this % y_validate )
        deallocate( this % y_validate_pre )
        
        deallocate( this % X_test )
        deallocate( this % y_test )    
        deallocate( this % y_test_pre ) 

		deallocate( this % X_batch )        
        deallocate( this % y_batch )
        deallocate( this % y_batch_pre )
        
        deallocate( this % my_NNTrain )
        deallocate( this % cross_entropy_function )
		deallocate( this % batch_generator )		
		deallocate( this % adam_method )
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* 析构函数，清理内存空间
    subroutine MNISTCase_clean_space( this )
    implicit none
        type(MNISTCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("MNISTCase: SUBROUTINE clean_space.")
        
        return
    end subroutine MNISTCase_clean_space
    !====
    
end module