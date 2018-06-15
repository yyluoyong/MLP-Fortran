!* ��ģ�鶨����������ѵ�������ݺͷ�����
!* ����Ĳ�������������μ�PDF�ĵ���   
module mod_NNTrain
use mod_Precision
use mod_ActivationFunctionList
use mod_BaseActivationFunction
use mod_NNStructure
use mod_Log
implicit none    

!---------------------------------
! �����ࣺ������ѵ���ķ��������� |
!---------------------------------
type, public :: NNTrain

    !* ���������������ֵ
    real(kind=PRECISION), public :: error_single
    !* ȫ��������ƽ�������ֵ
    real(kind=PRECISION), public :: error_avg
        
    !* ��������
    integer, public :: sample_count 
        
    ! �����Ŀ�����������
    integer, public :: layers_count    
        
    ! ÿ��ڵ���Ŀ���ɵ�����: 
    !     ����Ĵ�С�����в����Ŀ��������㣩
    integer, dimension(:), allocatable, public :: layers_node_count
    
    !* Ȩֵ��ѧϰ����
    real(kind=PRECISION), dimension(:), allocatable, public :: learning_rate_weight
    !* ��ֵ��ѧϰ����
    real(kind=PRECISION), dimension(:), allocatable, public :: learning_rate_threshold

    !* ѵ�����ݣ�ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X
    !* ѵ�����ݶ�Ӧ��Ŀ��ֵ��ÿһ����һ��
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y
    
    
    
    !* �����߱�ʶ�������ڶ�ȡָ����������Ϣ�ȡ�
    character(len=180), private :: caller_name = ''

    ! �Ƿ��ʼ����ɵı�ʶ
    logical, private :: is_init = .false.
        
    ! �Ƿ��ʼ���ڴ�ռ�
    logical, private :: is_allocate_done = .false.
    
    character(len=180), private :: NNParameter_path = &
        './ParameterSetting/'
    
    !* ���������Ϣ    
    character(len=180), private :: NNParameter_file = &
        './ParameterSetting/NNParameter.nml'
        
    !* ���ز�ÿ������Ŀ������
    character(len=180), private :: NNLayerNodeCount_file = &
        './ParameterSetting/NNHiddenLayerNodeCount.parameter'
        
    !* ÿ���Ȩֵѧϰ���ʡ���ֵѧϰ����
    character(len=180), private :: NNLearningRate_file = &
        './ParameterSetting/NNLearningRate.parameter'
    
    !* ÿ��ļ����s
    character(len=180), private :: NNActivationFunctionList_file = &
        './ParameterSetting/NNActivationFunctionList.parameter'    
        
    !* ������б�
    character(len=20), dimension(:), allocatable, private :: act_fun_name_list
        
    !* ѵ������
    integer, public :: train_step
    
    !* ʹ�õ�BPѵ���㷨
    character(len=20), private :: bp_algorithm
    
    !* ����ṹ
    type(NNStructure), pointer, private :: my_NNStructure
    
!||||||||||||    
contains   !|
!||||||||||||

    procedure, private :: init => m_init
    
    procedure, public :: train => m_train
    procedure, public :: sim   => m_sim

    procedure, private :: standard_BP_update     => m_standard_BP_update
    procedure, private :: accumulation_BP_update => m_accumulation_BP_update
    
    procedure, private :: get_total_error => m_get_total_error
    
    procedure, private :: allocate_memory   => m_allocate_memory
    procedure, private :: deallocate_memory => m_deallocate_memory
    
    procedure, private :: init_NNParameter                => m_init_NNParameter
    procedure, private :: load_NNParameter                => m_load_NNParameter
    procedure, private :: load_NNParameter_array          => m_load_NNParameter_array
    procedure, private :: load_NNActivation_Function_List => m_load_NNActivation_Function_List
    
    final :: NNTrain_clean_space
    
end type NNTrain
!===================

    !-------------------------
    private :: m_init
    private :: m_train
    private :: m_sim
    
    private :: m_standard_BP_update
    private :: m_accumulation_BP_update
    
    private :: m_get_total_error
    
    private :: m_allocate_memory
    private :: m_deallocate_memory
    
    private :: m_init_NNParameter
    private :: m_load_NNParameter
    private :: m_load_NNParameter_array
    private :: m_load_NNActivation_Function_List
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* ��ʼ����
    !* (1). ���ļ��ж�ȡ���������ѵ������;
    !* (2). �����ڴ�ռ�;
    !* (3). ��ʼ������ṹ.
    subroutine m_init( this, caller_name, X, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* ��������Ϣ��ֵ����Ϊ ''����ʱʹ��Ĭ��������Ϣ
        character(len=*), intent(in) :: caller_name
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: y

        class(BaseActivationFunction), pointer :: pt_act_fun
        type(ActivationFunctionList),  pointer :: pt_act_fun_list
        integer :: X_shape(2), Y_shape(2) 
        integer :: i
        
        if( .not. this % is_init ) then
        
            this % caller_name = caller_name
            call this % init_NNParameter(caller_name)
        
            !* ���ļ���ȡ������Ϣ
            call this % load_NNParameter()          
            call this % allocate_memory()
            call this % load_NNParameter_array()
            call this % load_NNActivation_Function_List()
            
            X_shape = SHAPE(X)
            Y_shape = SHAPE(y)
            
            !* ���������Ŀ
            this % layers_node_count(0) = X_shape(1)
            !* ���������Ŀ
            this % layers_node_count(this % layers_count) =  &
                Y_shape(1)
            !* ������Ŀ
            this % sample_count = X_shape(2)         
            
        
            !* ����ѵ����������Ŀ�꣬
            !*     ע������ָ������Ϊ�ڵ�ǰ���п��ܸı�X, y
            allocate( this % X, SOURCE=X )
            allocate( this % y, SOURCE=y )
            
            
            !* ��ʼ�� my_NNStructure
            allocate( this % my_NNStructure )
            
            allocate( pt_act_fun_list )
            
            call this % my_NNStructure % init_basic( &
                this % layers_count, this % layers_node_count)
        
            do i=1, this % layers_count
                call pt_act_fun_list % get_activation_function_by_name( &
                    this % act_fun_name_list(i), &
                    this % my_NNStructure % pt_Layer(i) % act_fun)           
            end do
                
            this % is_init = .true.
            
            call LogDebug("NNTrain: SUBROUTINE m_init")
            
        end if

        return
    end subroutine m_init
    !====

    !* ѵ������
    subroutine m_train( this, caller_name, X, t, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* ��������Ϣ��ֵ����Ϊ ''����ʱʹ��Ĭ��������Ϣ
        character(len=*), intent(in) :: caller_name
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(inout) :: y
        
        integer :: sample_index, t_step
        real(PRECISION) :: err, acc
        character(len=180) :: msg
        character(len=30) :: step_to_string, err_to_string, &
            acc_to_string
          
        !* ��ʼ��
        call this % init( caller_name, X, t )
        
        do t_step=1, this % train_step
        
            call LogDebug("NNTrain: SUBROUTINE m_train step")
            
            do sample_index=1, this % sample_count

                call this % my_NNStructure % backward_propagation( X(:, sample_index), &
                    t(:, sample_index), y(:, sample_index) )
        
                !* ��׼BP�㷨�ڴ˴���������Ȩֵ����ֵ
                if (TRIM(ADJUSTL(this % bp_algorithm)) == 'standard') then
                    call this % standard_BP_update()
                end if
            end do
            
            !* �ۻ�BP�㷨�ڴ˴���������Ȩֵ����ֵ 
            if (TRIM(ADJUSTL(this % bp_algorithm)) == 'accumulation') then
                    call this % accumulation_BP_update()
            end if
            
            call this % get_total_error(t, y, err)        
            call m_get_accuracy(t, y, acc)  
            
            write(UNIT=step_to_string, FMT='(I15)') t_step
            write(UNIT=err_to_string, FMT='(ES16.5)') err
            write(UNIT=acc_to_string, FMT='(F8.5)') acc
            msg = "t_step = " // TRIM(ADJUSTL(step_to_string)) // &
                ",  err = " // TRIM(ADJUSTL(err_to_string)) // &
                ",  acc = " // TRIM(ADJUSTL(acc_to_string))
            call LogInfo(msg)
            
            if (err < this % error_avg) then
                exit
            end if
            
            !call m_get_accuracy(t, y, err)  
            !
            !write(UNIT=step_to_string, FMT='(I15)') t_step
            !write(UNIT=err_to_string, FMT='(ES16.5)') err
            !msg = "t_step = " // TRIM(ADJUSTL(step_to_string)) // &
            !    ",  acc = " // TRIM(ADJUSTL(err_to_string))
            !call LogInfo(msg)
     
        end do
        
        return
    end subroutine m_train
    !====
    
    !* ��Ϻ���
    subroutine m_sim( this, X, t, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* X ������ֵ��t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(out) :: y
        
        integer :: sample_index
        real(PRECISION) :: err, acc
        
        if( .not. this % is_init ) then
            call LogErr("NNTrain: SUBROUTINE m_sim, &
                NNTrain need init first.")
        end if
        
        do sample_index=1, this % sample_count
            call this % my_NNStructure % forward_propagation( X(:, sample_index), &
                t(:, sample_index), y(:, sample_index) )
        end do
        
        !* undo���������е� t��Ԥ��� y���������ȵ�.
        !* call this % get_error(t, y)
        call m_get_accuracy(t, y, acc) 
        write(*, *) "Sim acc = ", acc
        
        return
    end subroutine m_sim
    !====
      
    !* �������
    subroutine m_get_total_error( this, t, y, err )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* t ��ʵ�������y ������Ԥ�����
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: err
        
        integer :: t_shape(2)   
        real(PRECISION) :: max_err
        
        t_shape = SHAPE(t)
        
        err = SUM(ABS(t - y))
        err = err / t_shape(2)
        
        max_err = MAXVAL(ABS(t - y))
        write(*, *) 'max_err = ', max_err
        
        call LogDebug("NNTrain: SUBROUTINE m_get_total_error")
             
        return
    end subroutine m_get_total_error
    !====
    
    !* ������ȷ��
    subroutine m_get_accuracy( t, y, acc )
    implicit none
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: acc
    
        integer :: y_shape(2), j, tag
        integer :: max_index_t(1), max_index_y(1)
        
        y_shape = SHAPE(y)
        
        tag = 0
        do j=1, y_shape(2)
            max_index_t = MAXLOC(t(:,j))
            max_index_y = MAXLOC(y(:,j))
            
            if (max_index_t(1) == max_index_y(1)) then
                tag = tag + 1
            end if
        end do
        
        acc = 1.0 * tag / y_shape(2)
        
        return
    end subroutine m_get_accuracy
    !====

    !* ��׼BP�㷨.
    subroutine m_standard_BP_update( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: layer_index, l_count
        real(PRECISION) :: eta_w, eta_theta
        
        l_count = this % layers_count
        
        do layer_index=1, l_count
        
            eta_w = this % learning_rate_weight(layer_index)
            eta_theta = this % learning_rate_threshold(layer_index)
            
            associate ( &
                W      => this % my_NNStructure % pt_W( layer_index ) % W,         &
                Theta  => this % my_NNStructure % pt_Theta( layer_index ) % Theta, &
                dW     => this % my_NNStructure % pt_Layer( layer_index ) % dW,    &               
                dTheta => this % my_NNStructure % pt_Layer( layer_index ) % dTheta &
            )
                
            !* W = W - �� * dW
            W = W - eta_w * dW
            
            !* �� = �� - �� * dTheta
            Theta = Theta -eta_theta * dTheta
           
            end associate
        end do
        
        return
    end subroutine m_standard_BP_update
    !====
    
    
    !* �ۻ�BP�㷨.
    subroutine m_accumulation_BP_update( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: layer_index, l_count
        real(PRECISION) :: eta_w, eta_theta
        
        l_count = this % layers_count     
        
        call LogDebug("NNTrain: SUBROUTINE m_train step")
        
        do layer_index=1, l_count
        
            eta_w = this % learning_rate_weight(layer_index)
            eta_theta = this % learning_rate_threshold(layer_index)
        
            associate ( &
                W          => this % my_NNStructure % pt_W( layer_index ) % W,             &
                Theta      => this % my_NNStructure % pt_Theta( layer_index ) % Theta,     &
                sum_dW     => this % my_NNStructure % pt_Layer( layer_index ) % sum_dW,    &              
                sum_dTheta => this % my_NNStructure % pt_Layer( layer_index ) % sum_dTheta &
            )
                      
            !* W = W - �� * �� dW
            W = W - eta_w * sum_dW

            !* �� = �� - �� * �� dTheta
            Theta = Theta - eta_theta * sum_dTheta
            
            !* ÿ����һ�ֱ����� 0
            sum_dW = 0
            sum_dTheta = 0
                
            end associate
        end do
        
        return
    end subroutine m_accumulation_BP_update
    !====  
    
    !* ��ʼ���������ļ�������·��
    subroutine m_init_NNParameter( this, caller_name )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* ��������Ϣ��ֵ����Ϊ ''����ʱʹ��Ĭ��������Ϣ
        character(len=*), intent(in) :: caller_name
        
        if (caller_name /= '') then
            this % NNParameter_file = &
                TRIM(ADJUSTL(this % NNParameter_path)) // &
                TRIM(ADJUSTL(caller_name)) // '_' // &
                'NNParameter.nml'
                
            this % NNLayerNodeCount_file = &
                TRIM(ADJUSTL(this % NNParameter_path)) // &
                TRIM(ADJUSTL(caller_name)) // '_' // &
                'NNHiddenLayerNodeCount.parameter'
                
            this % NNLearningRate_file = &
                TRIM(ADJUSTL(this % NNParameter_path)) // &
                TRIM(ADJUSTL(caller_name)) // '_' // &
                'NNLearningRate.parameter'
                
            this % NNActivationFunctionList_file = &
                TRIM(ADJUSTL(this % NNParameter_path)) // &
                TRIM(ADJUSTL(caller_name)) // '_' // &
                'NNActivationFunctionList.parameter'
        end if
    
        call LogDebug("NNTrain: SUBROUTINE m_init_NNParameter")
        
        return
    end subroutine
    !====
    
    
    !* ��ȡ����Ĳ���
    subroutine m_load_NNParameter( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: HIDDEN_LAYERS_COUNT, TRAIN_STEP
        real(PRECISION) :: ERROR_SINGLE, ERROR_AVG
        character(len=20) :: BP_ALGORITHM
        namelist / NNParameter_NameList / HIDDEN_LAYERS_COUNT, &
            TRAIN_STEP, ERROR_SINGLE, ERROR_AVG, &
            BP_ALGORITHM
            
        integer :: l_count  
        
        !* ��ȡ������Ϣ���������ز������
        open( UNIT=30, FILE=this % NNParameter_file, &
            form='formatted', status='old' )            
        read( unit=30, nml=NNParameter_NameList )        
        close(unit=30)
        
        l_count = HIDDEN_LAYERS_COUNT + 1
        this % layers_count = l_count
        this % train_step = TRAIN_STEP
        
        this % error_single = ERROR_SINGLE
        this % error_avg = ERROR_AVG

		this % bp_algorithm = TRIM(ADJUSTL(BP_ALGORITHM))   
        
        call LogDebug("NNTrain: SUBROUTINE m_load_NNParameter")
        
        return
    end subroutine m_load_NNParameter
    !====

    !* ��ȡ����Ĳ���
    subroutine m_load_NNParameter_array( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: l_count, hidden_l_count
        
        l_count = this % layers_count
        hidden_l_count = l_count - 1
        
        !* ��ȡÿ�����ز�Ľ����Ŀ
        open( UNIT=30, FILE=this % NNLayerNodeCount_file, &
            form='formatted', status='old' )            
        read( 30, * ) this % layers_node_count(1:hidden_l_count)       
        close(unit=30)
        
        !* ��ȡȨֵ����ֵѧϰ��������
        open( UNIT=30, FILE=this % NNLearningRate_file, &
            form='formatted', status='old' )            
        read( 30, * ) this % learning_rate_weight 
        read( 30, * ) this % learning_rate_threshold
        close(unit=30)
        
        call LogDebug("NNTrain: SUBROUTINE m_load_NNParameter_array")
		   
        return
    end subroutine m_load_NNParameter_array
    !====
    
    !* ��ȡ���㼤�������
    subroutine m_load_NNActivation_Function_List( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: l_count
        integer :: i
        
        l_count = this % layers_count
        
        !* ��ȡÿ�����ز�Ľ����Ŀ
        open( UNIT=30, FILE=this % NNActivationFunctionList_file, &
            form='formatted', status='old' )  
            
        do i=1, l_count
            read( 30, * ) this % act_fun_name_list(i)  
        end do
        
        do i=1, l_count
            write( *, * ) this % act_fun_name_list(i)  
        end do
        
        close(unit=30)
        
        call LogDebug("NNTrain: SUBROUTINE m_load_NNActivation_Function_List")
		   
        return
    end subroutine m_load_NNActivation_Function_List
    !====
    
    !* �����ڴ�ռ�
    subroutine m_allocate_memory( this )
    implicit none
        class(NNTrain), intent(inout) :: this
		
        integer :: l_count
        
        l_count = this % layers_count
        
        allocate( this % layers_node_count(0:l_count) )        
        allocate( this % learning_rate_weight(l_count) )
        allocate( this % learning_rate_threshold(l_count) )
        allocate( this % act_fun_name_list(l_count) )       
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* �����ڴ�ռ�
    subroutine m_deallocate_memory( this )
    implicit none
        class(NNTrain), intent(inout)  :: this	
        
        deallocate(this % layers_node_count)
        deallocate(this % learning_rate_weight)
        deallocate(this % learning_rate_threshold)
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* ���������������ڴ�ռ�
    subroutine NNTrain_clean_space( this )
    implicit none
        type(NNTrain), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("NNTrain: SUBROUTINE clean_space.")
        
        return
    end subroutine NNTrain_clean_space
    !====

end module