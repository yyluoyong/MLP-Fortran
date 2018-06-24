module mod_MNISTCase
use mod_Precision
use mod_Log
use mod_BaseCalculationCase
use mod_NNTrain
use mod_CrossEntropy
use mod_SimpleBatchGenerator
implicit none    

!------------------------------
! �����ࣺMNIST���ݼ��������� |
!------------------------------
type, extends(BaseCalculationCase), public :: MNISTCase
    !* �̳���BaseCalculationCase��ʵ����ӿ�
    
    character(len=180), private :: train_image_data_file = &
        './Data/MNISTCase/train-images.fortran'
    character(len=180), private :: train_label_data_file = &
        './Data/MNISTCase/train-labels.fortran'
    character(len=180), private :: test_image_data_file = &
        './Data/MNISTCase/t10k-images.fortran'
    character(len=180), private :: test_label_data_file = &
        './Data/MNISTCase/t10k-labels.fortran'

    !* �Ƿ��ʼ���ڴ�ռ�
    logical, private :: is_allocate_done = .false.
	
	!* ÿ������������
    integer, public :: batch_size = 100
    
    !* ѵ�������������������60000
    integer, public :: count_train_sample = 60000
    
    !* ���Լ����������������10000
    integer, public :: count_test_sample = 5000
    
    !* ����������������: 28 ��28 = 784
    integer, public :: sample_point_X = 784
    integer, public :: sample_point_y = 10
	
	!* ѵ�����ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_batch
    !* ѵ�����ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch
    !* ѵ�����ݵ�Ԥ����
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_batch_pre
    
    !* ѵ�����ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_train
    !* ѵ�����ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train
    !* ѵ�����ݵ�Ԥ����
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_train_pre
    
    !* �������ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X_test
    !* �������ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test
    !* �������ݵ�Ԥ����
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y_test_pre
    
    type(NNTrain), pointer :: my_NNTrain
    
    type(CrossEntropyWithSoftmax), pointer :: cross_entropy_function
	
	type(SimpleBatchGenerator), pointer :: batch_generator
    
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

    !* ������
    subroutine m_main( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
		
		integer :: train_count = 10000
        integer :: round_step
        character(len=20) :: round_step_to_str
    
        call this % allocate_memory()
        
        call this % load_MNIST_data()
        
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
        
        X_train = ( X_train ) / 128.0 - 1
        
        call this % my_NNTrain % init('MNISTCase', X_train, y_train)
        
        call this % my_NNTrain % set_train_type('classification')
        
        call this % my_NNTrain % &
            set_weight_threshold_init_methods_name('xavier')
            
        call this % my_NNTrain % set_loss_function(this % cross_entropy_function)
		
		!do round_step=1, train_count     
  !          
  !          call this % batch_generator % get_next_batch( &
  !              X_train, y_train, X_batch, y_batch )
  !          
  !          call this % my_NNTrain % train(X_batch, &
  !              y_batch, y_batch_pre)    
  !          
  !          write(UNIT=round_step_to_str, FMT='(I15)') round_step
  !          call LogInfo("round_step = " // TRIM(ADJUSTL(round_step_to_str)))
  !          
  !          !call this % my_NNTrain % sim(X_train, &
  !          !    y_train, y_train_pre)
  !          
  !          !if (MOD(round_step, 1) == 0) then
  !              call this % my_NNTrain % sim(X_test, &
  !                  y_test, y_test_pre)
  !          !end if
  !          
  !      end do
        
        !call this % my_NNTrain % sim(X_train, &
        !    y_train, y_train_pre)
        !
        call this % my_NNTrain % train(X_train, &
            y_train, y_train_pre)
        
        call this % my_NNTrain % sim(X_test, &
            y_test, y_test_pre)
        
            
        end associate
            
        return
    end subroutine m_main
    !====
    
    !* ��ȡMNIST����
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
    
    !* ���ļ��ж�ȡMNIST����
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
            
            !* ��ȡ label
            read(30) magic_number, sample_count
            
            if (magic_number /= 2049) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file &
                    --> magic_number /= 2049.")
                stop
            end if
            
            read(30) (data_array_int4(1,j), j=1, data_shape(2)) 

            !* label��ȡֵ��Χ�ǣ�0-9
            !* ��data_arrayת����one-hot��ʽ������
            !* label = 0 --> [1,0,0,0,0,0,0,0,0,0]
            !* label = 1 --> [0,1,0,0,0,0,0,0,0,0]
            !* �Դ����� ... 
            !* label = 9 --> [0,0,0,0,0,0,0,0,0,1]
            data_array = 0
            do j=1, data_shape(2)
                data_array(data_array_int4(1,j)+1, j) = 1.0
            end do
            
            deallocate( data_array_int4 )
            
        else if (data_shape(1) == this % sample_point_X) then   
        
            allocate( data_array_int4(data_shape(1), data_shape(2)) )
        
            !* ��ȡ image
            read(30) magic_number, sample_count, row, column
            
            if (magic_number /= 2051) then
                call LogErr("MNISTCase: SUBROUTINE m_read_MNIST_data_from_file &
                    --> magic_number /= 2051.")
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

    !* �����ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout) :: this
        
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
    

    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(MNISTCase), intent(inout)  :: this	
        
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

    
    !* ���������������ڴ�ռ�
    subroutine MNISTCase_clean_space( this )
    implicit none
        type(MNISTCase), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("MNISTCase: SUBROUTINE clean_space.")
        
        return
    end subroutine MNISTCase_clean_space
    !====
    
end module