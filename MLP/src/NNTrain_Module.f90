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

    ! 层的数目，不含输入层
    integer, public :: layers_count    
        
    ! 每层节点数目构成的数组: 
    !     数组的大小是所有层的数目（含输入层）
    integer, dimension(:), allocatable, public :: layers_node_count
	
	!* 网络结构
    type(NNStructure), pointer, public :: my_NNStructure
	
    !* 默认为空，即使用NNStructure中定义的(-1,1)
    character(len=30), private :: weight_threshold_init_methods_name = ''
    
    !* 调用者标识，可用于读取指定的配置信息等。
    character(len=180), private :: caller_name = ''

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
        
    !* 内置的训练步数计数器
    integer, private :: train_step_counter = 0
    
    !* 使用的BP训练算法
    character(len=20), private :: bp_algorithm
       
	!* 优化方法
	class(BaseGradientOptimizationMethod), pointer, private :: gradient_optimization_method
	
!||||||||||||    
contains   !|
!||||||||||||

    procedure, public :: init => m_init
    
    procedure, public :: set_caller_name   => m_set_caller_name
    procedure, public :: set_loss_function => m_set_loss_function
    procedure, public :: set_weight_threshold_init_methods_name => &
        m_set_weight_threshold_init_methods_name
	procedure, public :: set_optimization_method => m_set_optimization_method
    
    procedure, public :: train => m_train
    procedure, public :: sim   => m_sim
    
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
    private :: m_set_weight_threshold_init_methods_name
    private :: m_set_loss_function
	private :: m_set_optimization_method
	
    private :: m_init_NNParameter
    private :: m_load_NNParameter
    private :: m_load_NNParameter_array
    private :: m_load_NNActivation_Function_List
	
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
        
        integer :: sample_index
        integer :: X_shape(2)
        
		associate (                                                 &
            step            => this % train_step_counter,           &
            grad_opt_method => this % gradient_optimization_method, &
            my_NN           => this % my_NNStructure,               &
            bp_algorithm    => this % bp_algorithm                  &           
        )   
		
		step = step + 1
		
        X_shape = SHAPE(X)        
        
		!* Some gradient optimization methods need preprocessing 
		!* before start iteration, like Adam e.t.c.
        call grad_opt_method % pre_process()
		!* Some gradient optimization methods need the iteration step value 
		!* to adjustment the algorithm intrinsic parameter 
		!* before start iteration, like Adam e.t.c.		
		call grad_opt_method % set_iterative_step( step )
            
        do sample_index=1, X_shape(2)

            call my_NN % backward_propagation( X(:, sample_index), &
                t(:, sample_index), y(:, sample_index) )
        
            !* 标准BP算法在此处更新网络权值和阈值
            if (TRIM(ADJUSTL(bp_algorithm)) == 'standard') then
                call grad_opt_method % update_NN(bp_algorithm)
            end if
				
			call my_NN % calc_average_gradient( X_shape(2) )
			
        end do
            
        !* 累积BP算法在此处更新网络权值和阈值 
        if (TRIM(ADJUSTL(bp_algorithm)) == 'accumulation') then
			call grad_opt_method % update_NN(bp_algorithm)
        end if
            
		call grad_opt_method % post_process()
            
		call my_NN % set_average_gradient_zero()
        
		end associate
		
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
        
        if( .not. this % is_init ) then
            call LogErr("NNTrain: SUBROUTINE m_sim, &
                NNTrain need init first.")
        end if
        
        X_shape = SHAPE(X)
        
        do sample_index=1, X_shape(2)
            call this % my_NNStructure % forward_propagation( X(:, sample_index), &
                t(:, sample_index), y(:, sample_index) )
        end do
        
        return
    end subroutine m_sim
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
	
    !* 读取网络的参数
    subroutine m_load_NNParameter( this )
    implicit none
        class(NNTrain), intent(inout) :: this
        
        integer :: HIDDEN_LAYERS_COUNT
        character(len=20) :: BP_ALGORITHM
        namelist / NNParameter_NameList / HIDDEN_LAYERS_COUNT, BP_ALGORITHM
            
        integer :: l_count  
        
        !* 读取参数信息，比如隐藏层的数量
        open( UNIT=30, FILE=this % NNParameter_file, &
            form='formatted', status='old' )            
        read( unit=30, nml=NNParameter_NameList )        
        close(unit=30)
        
        l_count = HIDDEN_LAYERS_COUNT + 1
        this % layers_count = l_count
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