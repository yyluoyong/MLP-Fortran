!* 该模块定义了神经网络训练的数据和方法。
!* 具体的参数、函数意义参见PDF文档。   
module mod_NNTrain
use mod_Precision
use mod_ActivationFunctionList
use mod_BaseActivationFunction
use mod_NNStructure
use mod_Log
use mod_BaseGradientOptimizationMethod
use mod_NNTools
use mod_Tools
implicit none    

!-----------------------------------
! 工作类：神经网络训练的方法和数据 |
!-----------------------------------
type, public :: NNTrain

    !* 单个样本的误差阈值
    real(kind=PRECISION), public :: error_single
    !* 全部样本的平均误差阈值
    real(kind=PRECISION), public :: error_avg

    ! 层的数目，不含输入层
    integer, public :: layers_count    
        
    ! 每层节点数目构成的数组: 
    !     数组的大小是所有层的数目（含输入层）
    integer, dimension(:), allocatable, public :: layers_node_count
	
    !* 默认为空，即使用NNStructure中定义的(-1,1)
    character(len=30), private :: weight_threshold_init_methods_name = ''
    
    !* 调用者标识，可用于读取指定的配置信息等。
    character(len=180), private :: caller_name = ''

    !* 问题的类别标识
    !* regression 回归问题，默认值
    !* classification 分类问题
    character(len=20), private :: train_type = 'regression'
    
    ! 是否初始化完成的标识
    logical, private :: is_init = .false.
        
    ! 是否初始化内存空间
    logical, private :: is_allocate_done = .false.
    
    character(len=180), private :: NNParameter_path = &
        './ParameterSetting/'
    
    !* 网络参数信息    
    character(len=180), private :: NNParameter_file = &
        './ParameterSetting/NNParameter.nml'
        
    !* 隐藏层每层结点数目的数组
    character(len=180), private :: NNLayerNodeCount_file = &
        './ParameterSetting/NNHiddenLayerNodeCount.parameter'
    
    !* 每层的激活函数s
    character(len=180), private :: NNActivationFunctionList_file = &
        './ParameterSetting/NNActivationFunctionList.parameter'    
        
    !* 激活函数列表
    character(len=20), dimension(:), allocatable, private :: act_fun_name_list
        
    !* 训练步数
    integer, public :: train_step
	
	!* 训练过程中信息输出间隔步数
	integer, public :: train_msg_output_step = 500
    
    !* 使用的BP训练算法
    character(len=20), private :: bp_algorithm
    
    !* 网络结构
    type(NNStructure), pointer, public :: my_NNStructure
    
	!* 优化方法
	class(BaseGradientOptimizationMethod), pointer, private :: gradient_optimization_method
	
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: init => m_init
    
    procedure, public :: set_caller_name   => m_set_caller_name
	procedure, public :: set_train_type    => m_set_train_type
    procedure, public :: set_loss_function => m_set_loss_function
    procedure, public :: set_weight_threshold_init_methods_name => &
        m_set_weight_threshold_init_methods_name
	procedure, public :: set_optimization_method => m_set_optimization_method
    procedure, public :: set_train_msg_output_step => m_set_train_msg_output_step
    
    procedure, public :: train => m_train
    procedure, public :: sim   => m_sim

    procedure, private :: step_post_process => m_step_post_process
    
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

    private :: m_set_caller_name
    private :: m_set_train_type
    private :: m_set_weight_threshold_init_methods_name
    private :: m_set_loss_function
	private :: m_set_optimization_method
	private :: m_set_train_msg_output_step
    
    private :: m_init_NNParameter
    private :: m_load_NNParameter
    private :: m_load_NNParameter_array
    private :: m_load_NNActivation_Function_List
    
    private :: m_step_post_process
	
	private :: m_allocate_memory
    private :: m_deallocate_memory
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* 初始化：
    !* (1). 从文件中读取网络参数、训练参数;
    !* (2). 申请内存空间;
    !* (3). 初始化网络结构.
    subroutine m_init( this, caller_name, count_input_node, count_output_node )
    use mod_NNWeightThresholdInitMethods
    implicit none
        class(NNTrain), intent(inout) :: this
        character(len=*), intent(in) :: caller_name
        integer, intent(in) :: count_input_node, count_output_node

        class(BaseActivationFunction), pointer :: pt_act_fun
        type(ActivationFunctionList),  pointer :: pt_act_fun_list
        integer :: i
        
        if( .not. this % is_init ) then
        
            this % caller_name = caller_name
            call this % init_NNParameter(caller_name)
        
            !* 从文件读取参数信息
            call this % load_NNParameter()          
            call this % allocate_memory()
            call this % load_NNParameter_array()
            call this % load_NNActivation_Function_List()
            
			associate (                                       &
				layers_count      => this % layers_count,     &
                layers_node_count => this % layers_node_count &	
			)   
			
            !* 输入层结点数目
            layers_node_count(0) = count_input_node
            !* 输出层结点数目
            layers_node_count(layers_count) = count_output_node                
            
            !* 初始化 my_NNStructure
            allocate( this % my_NNStructure )
            
            allocate( pt_act_fun_list )
            
            call this % my_NNStructure % init_basic( layers_count,layers_node_count)
        
			!* 给每层设置激活函数
            do i=1, this % layers_count
                call pt_act_fun_list % get_activation_function_by_name( &
                    this % act_fun_name_list(i),                        &
                    this % my_NNStructure % pt_Layer(i) % act_fun)           
            end do
            
			!* 给每层权值、阈值按指定方式初始化
            call NN_weight_threshold_init_main(            &
                this % weight_threshold_init_methods_name, &
                this % my_NNStructure)
                
            this % is_init = .true.
            
			end associate
			
            call LogDebug("NNTrain: SUBROUTINE m_init")
            
        end if

        return
    end subroutine m_init
    !====

    !* 训练函数
    subroutine m_train( this, X, t, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* X 是输入值，t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(inout) :: y
        
        integer :: sample_index, t_step
        integer :: X_shape(2)
        real(PRECISION) :: err, max_err, acc
        character(len=20) :: step_to_string
        character(len=180) :: msg
        
        X_shape = SHAPE(X)        
        
        call this % gradient_optimization_method % pre_process()
        
        do t_step=1, this % train_step
        
            call LogDebug("NNTrain: SUBROUTINE m_train step")
						
			call this % gradient_optimization_method % set_iterative_step( t_step )
            
            do sample_index=1, X_shape(2)

                call this % my_NNStructure % backward_propagation( X(:, sample_index), &
                    t(:, sample_index), y(:, sample_index) )
        
                !* 标准BP算法在此处更新网络权值和阈值
                if (TRIM(ADJUSTL(this % bp_algorithm)) == 'standard') then
                    call this % gradient_optimization_method % &
						update_NN(this % bp_algorithm)
                end if
				
				call this % my_NNStructure % calc_average_gradient( X_shape(2) )
            end do
            
            !* 累积BP算法在此处更新网络权值和阈值 
            if (TRIM(ADJUSTL(this % bp_algorithm)) == 'accumulation') then
				call this % gradient_optimization_method % &
					update_NN(this % bp_algorithm)
            end if
            
			call this % gradient_optimization_method % post_process()
			
            call this % step_post_process(t_step, t, y, err, max_err, acc)
            
			call this % my_NNStructure % set_average_gradient_zero()
			
            if (err < this % error_avg) then
            !if (err < this % error_single) then
                write(UNIT=step_to_string, FMT='(I15)') t_step
                call LogInfo("--> step_end = " // TRIM(ADJUSTL(step_to_string)))
                exit
            end if	
   
        end do
        
        return
    end subroutine m_train
    !====
    
    !* 拟合函数
    subroutine m_sim( this, X, t, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* X 是输入值，t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(out) :: y
        
        integer :: sample_index
        integer :: X_shape(2)
        real(PRECISION) :: err, acc
        character(len=20) :: acc_to_string
        character(len=180) :: msg
        
        if( .not. this % is_init ) then
            call LogErr("NNTrain: SUBROUTINE m_sim, &
                NNTrain need init first.")
        end if
        
        X_shape = SHAPE(X)
        
        do sample_index=1, X_shape(2)
            call this % my_NNStructure % forward_propagation( X(:, sample_index), &
                t(:, sample_index), y(:, sample_index) )
        end do
        
        !* undo：根据已有的 t，预测的 y，计算误差等等.
        !* call this % get_error(t, y)
        call calc_classify_accuracy(t, y, acc) 
        
        write(UNIT=acc_to_string, FMT='(F8.5)') acc
        msg = "Sim acc = " // TRIM(ADJUSTL(acc_to_string))
        call LogInfo(msg)
        
        return
    end subroutine m_sim
    !====
    
    
    !* 获取单步迭代的误差或者精确度等并输出显示信息
    subroutine m_step_post_process( this, step, t, y, err, max_err, acc )
    implicit none
        class(NNTrain), intent(inout) :: this
        integer, intent(in) :: step
        !* t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: err
        real(PRECISION), optional, intent(inout) :: acc, max_err
    
        real(PRECISION) :: acc_local
        character(len=20) :: step_to_string, err_to_string,  &
            max_err_to_string, acc_to_string
        character(len=180) :: msg        
		
		select case (TRIM(ADJUSTL(this % train_type)))
        case ('regression')
            call calc_L_2_error(t, y, err)
			call calc_L_inf_error(t, y, max_err)
        case ('classification')
            call calc_cross_entropy_error( t, y, err, max_err )
        case default
            call LogErr("NNTrain: SUBROUTINE m_step_post_process, &
                train_type error.")
            stop       
        end select
        
        write(UNIT=step_to_string, FMT='(I15)') step  
        write(UNIT=err_to_string, FMT='(ES16.5)') err
        write(UNIT=max_err_to_string, FMT='(ES16.5)') max_err                   
               
        msg = "t_step = " // TRIM(ADJUSTL(step_to_string)) // &
            ",  err = " // TRIM(ADJUSTL(err_to_string)) // &
            ",  max_err = " // TRIM(ADJUSTL(max_err_to_string))
        
        select case (TRIM(ADJUSTL(this % train_type)))
        case ('regression')
            continue
        case ('classification')
            call calc_classify_accuracy( t, y, acc_local )
            if (PRESENT(acc)) then
                acc = acc_local        
            end if
            write(UNIT=acc_to_string, FMT='(F8.5)') acc_local
            msg = TRIM(ADJUSTL(msg)) // ", acc = " // &
                TRIM(ADJUSTL(acc_to_string))     
        end select
        
        if (MOD(step, this % train_msg_output_step) == 0) then
            call LogInfo(msg)
        end if
        
        call LogDebug("NNTrain: SUBROUTINE m_step_post_process")
        
        return
    end subroutine m_step_post_process
    !====  

    !* 初始化各参数文件的完整路径
    subroutine m_init_NNParameter( this, caller_name )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* 调用者信息，值可以为 ''，此时使用默认配置信息
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

            this % NNActivationFunctionList_file = &
                TRIM(ADJUSTL(this % NNParameter_path)) // &
                TRIM(ADJUSTL(caller_name)) // '_' // &
                'NNActivationFunctionList.parameter'
        end if
    
        call LogDebug("NNTrain: SUBROUTINE m_init_NNParameter")
        
        return
    end subroutine
    !====
    
    !* 设置权值、阈值处理方法的名字
    subroutine m_set_caller_name( this, caller_name )
    implicit none
        class(NNTrain), intent(inout) :: this
        character(len=*), intent(in) :: caller_name
    
        this % caller_name = caller_name
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_set_caller_name")
        
        return
    end subroutine m_set_caller_name
    !====
    
    !* 设置权值、阈值处理方法的名字
    subroutine m_set_weight_threshold_init_methods_name( this, name )
    use mod_NNWeightThresholdInitMethods
    implicit none
        class(NNTrain), intent(inout) :: this
        character(len=*), intent(in) :: name
    
        this % weight_threshold_init_methods_name = name
		
		call NN_weight_threshold_init_main(            &
            this % weight_threshold_init_methods_name, &
            this % my_NNStructure)
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE set_weight_threshold_init_methods_name.")
        
        return
    end subroutine
    !====
    
    !* 设置训练问题类别
    subroutine m_set_train_type( this, type_name )
    implicit none
        class(NNTrain), intent(inout) :: this
        character(len=*), intent(in) :: type_name
    
        this % train_type = type_name
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_set_train_type")
        
        return
    end subroutine m_set_train_type
    !====
 
    !* 设置激活函数
    subroutine m_set_loss_function( this, loss_fun )
    implicit none
        class(NNTrain), intent(inout) :: this
        class(BaseLossFunction), target, intent(in) :: loss_fun
        
        call this % my_NNStructure % set_loss_function( loss_fun )
        
        call LogDebug("NNTrain: SUBROUTINE m_set_loss_function")
        
        return
    end subroutine m_set_loss_function
    !====
    
	
	!* 设置优化方法
    subroutine m_set_optimization_method( this, opt_method )
    implicit none
        class(NNTrain), intent(inout) :: this
        class(BaseGradientOptimizationMethod), target, intent(in) :: opt_method
        
        this % gradient_optimization_method => opt_method
        
        call LogDebug("NNTrain: SUBROUTINE m_set_optimization_method")
        
        return
    end subroutine m_set_optimization_method
    !====	
	
	!* 设置优化方法
    subroutine m_set_train_msg_output_step( this, msg_step )
    implicit none
        class(NNTrain), intent(inout) :: this
        integer, intent(in) :: msg_step
        
        this % train_msg_output_step = msg_step
        
        call LogDebug("NNTrain: SUBROUTINE m_set_train_msg_output_step")
        
        return
    end subroutine m_set_train_msg_output_step
    !====
	
    !* 读取网络的参数
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
        
        !* 读取参数信息，比如隐藏层的数量
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

    !* 读取网络的参数
    subroutine m_load_NNParameter_array( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: l_count, hidden_l_count
        
        l_count = this % layers_count
        hidden_l_count = l_count - 1
        
        !* 读取每个隐藏层的结点数目
        open( UNIT=30, FILE=this % NNLayerNodeCount_file, &
            form='formatted', status='old' )            
        read( 30, * ) this % layers_node_count(1:hidden_l_count)       
        close(unit=30)
        
        call LogDebug("NNTrain: SUBROUTINE m_load_NNParameter_array")
		   
        return
    end subroutine m_load_NNParameter_array
    !====
    
    !* 读取各层激活函数名字
    subroutine m_load_NNActivation_Function_List( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: l_count
        integer :: i
        character(len=180) :: msg
        character(len=20) :: index_to_string
        
        l_count = this % layers_count
        
        !* 读取每个隐藏层的结点数目
        open( UNIT=30, FILE=this % NNActivationFunctionList_file, &
            form='formatted', status='old' )  
            
        do i=1, l_count
            read( 30, * ) this % act_fun_name_list(i)  
        end do
        
        call LogInfo("Activation Function List: ")
        do i=1, l_count    
            write(UNIT=index_to_string, FMT='(I15)') i
            msg = "--> layer index = " // TRIM(ADJUSTL(index_to_string)) // &
                ", activation function = " // &
                TRIM(ADJUSTL(this % act_fun_name_list(i)))
            call LogInfo(msg)
        end do
        
        close(unit=30)
        
        call LogDebug("NNTrain: SUBROUTINE m_load_NNActivation_Function_List")
		   
        return
    end subroutine m_load_NNActivation_Function_List
    !====
    
    !* 申请内存空间
    subroutine m_allocate_memory( this )
    implicit none
        class(NNTrain), intent(inout) :: this
		
        integer :: l_count
        
        l_count = this % layers_count
        
        allocate( this % layers_node_count(0:l_count)     ) 
        allocate( this % act_fun_name_list(l_count)       )       
        
        this % is_allocate_done = .true.
        
        call LogDebug("NNTrain: SUBROUTINE m_allocate_memory")
        
        return
    end subroutine m_allocate_memory
    !====
    

    !* 销毁内存空间
    subroutine m_deallocate_memory( this )
    implicit none
        class(NNTrain), intent(inout)  :: this	
        
        deallocate( this % layers_node_count       )
		deallocate( this % act_fun_name_list       )  
        
        this % is_allocate_done = .false.
        
        return
    end subroutine m_deallocate_memory 
    !====

    
    !* 析构函数，清理内存空间
    subroutine NNTrain_clean_space( this )
    implicit none
        type(NNTrain), intent(inout) :: this
    
        call this % deallocate_memory()
        
        call LogInfo("NNTrain: SUBROUTINE clean_space.")
        
        return
    end subroutine NNTrain_clean_space
    !====

end module