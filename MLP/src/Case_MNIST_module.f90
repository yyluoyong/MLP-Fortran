module mod_MNISTCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
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
    
    !* 训练集样本数量，最大是60000
    integer, public :: count_train_sample = 100
    
    !* 测试集样本数量，最大是10000
    integer, public :: count_test_sample = 5000
    
    !* 单个样本的数据量: 28 ×28 = 784
    integer, public :: sample_point_X = 784
    integer, public :: sample_point_y = 1
    
    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    
    !* 测试数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* 测试数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    
    type(NNTrain), pointer :: my_NNTrain
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: main      => m_main

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
    
        real(kind=PRECISION), dimension(:,:), allocatable :: y
    
        call this % allocate_memory()
        
        allocate( y, SOURCE = this % y_train)
        
        call this % load_MNIST_data()
        
        this % X_train = ( this % X_train ) / 256.0
        this % y_train = ( this % y_train ) / 9.0
        
        call this % my_NNTrain % train('MNISTCase', this % X_train, &
            this % y_train, y)
        
        return
    end subroutine m_main
    !====
    
    !* 读取MNIST数据
    subroutine m_load_MNIST_data( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
    
        call this % read_MNIST_data_from_file(&
            this % train_image_data_file, this % X_train)
        
        call this % read_MNIST_data_from_file(&
            this % train_label_data_file, this % y_train)
        
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
        allocate( data_array_int4(data_shape(1), data_shape(2)) )
        
        open(UNIT=30, FILE=file_name, &
            ACCESS='stream', FORM='unformatted', STATUS='old')
        
        if (data_shape(1) == this % sample_point_y) then 
            !* 读取 label
            read(30) magic_number, sample_count
            
            if (magic_number /= 2049) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file &
                    --> magic_number /= 2049.")
                stop
            end if
            
            read(30) ((data_array_int4(i,j), i=1, data_shape(1)), j=1, data_shape(2)) 
            
        else if (data_shape(1) == this % sample_point_X) then 
            !* 读取 image
            read(30) magic_number, sample_count, row, column
            
            if (magic_number /= 2051) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file &
                    --> magic_number /= 2051.")
                stop
            end if
            
            read(30) ((data_array_int4(i,j), i=1, data_shape(1)), j=1, data_shape(2)) 
        
        else
            call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file.")
            stop
        end if

        close(30)
        
        data_array = data_array_int4
        
        deallocate( data_array_int4 )
        
        return
    end subroutine m_read_MNIST_data_from_file
    !====

    !* 申请内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
        
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
    

    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout)  :: this	
        
        deallocate( this % X_train )        
        deallocate( this % y_train )
        
        deallocate( this % X_test )
        deallocate( this % y_test )    
        
        deallocate( this % my_NNTrain )
        
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