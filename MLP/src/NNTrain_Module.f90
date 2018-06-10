!* 该模块定义了神经网络训练的数据和方法。
!* 具体的参数、函数意义参见PDF文档。   
module mod_NNTrain
use mod_Precision
use mod_ActivationFunctionList
use mod_BaseActivationFunction
use mod_NNStructure
use mod_Log
implicit none    

!---------------------------------
! 工作类：神经网络训练的方法和数据 |
!---------------------------------
type, public :: NNTrain

    ! 是否初始化完成的标识
    logical, private :: is_init = .false.
        
    ! 是否初始化内存空间
    logical, private :: is_allocate_done = .false.
    
    !* 网络参数信息    
    character(len=180), private :: NNParameter_file = &
        './ParameterSetting/NNParameter.nml'
        
    !* 隐藏层每层结点数目的数组
    character(len=180), private :: NNLayerNodeCount_file = &
        './ParameterSetting/NNHiddenLayerNodeCount.parameter'
        
    !* 每层的权值学习速率、阈值学习速率
    character(len=180), private :: NNLearningRate_file = &
        './ParameterSetting/NNLearningRate.parameter'
    
    !* 每层的激活函数s
    character(len=180), private :: NNActivationFunctionList_file = &
        './ParameterSetting/NNActivationFunctionList.parameter'    
    
    !* 激活函数列表
    character(len=20), dimension(:), allocatable, private :: act_fun_name_list
        
    !* 训练步数
    integer, public :: train_step
    
    !* 使用的BP训练算法
    character(len=20), private :: bp_algorithm
    
    !* 单个样本的误差阈值
    real(kind=PRECISION), public :: error_single
    !* 全部样本的平均误差阈值
    real(kind=PRECISION), public :: error_avg
        
    !* 样本数量
    integer, public :: sample_count 
        
    ! 层的数目，不含输入层
    integer, public :: layers_count    
        
    ! 每层节点数目构成的数组: 
    !     数组的大小是所有层的数目（含输入层）
    integer, dimension(:), allocatable, public :: layers_node_count
    
    !* 权值的学习速率
    real(kind=PRECISION), dimension(:), allocatable, public :: learning_rate_weight
    !* 阈值的学习速率
    real(kind=PRECISION), dimension(:), allocatable, public :: learning_rate_threshold

    !* 训练数据，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: X
    !* 训练数据对应的目标值，每一列是一组
    real(kind=PRECISION), dimension(:,:), allocatable, public :: y
    
    !* 网络结构
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
    
    private :: m_load_NNParameter
    private :: m_load_NNParameter_array
    private :: m_load_NNActivation_Function_List
    !-------------------------
	
!||||||||||||    
contains   !|
!||||||||||||

    !* 初始化：
    !* (1). 从文件中读取网络参数、训练参数;
    !* (2). 申请内存空间;
    !* (3). 初始化网络结构.
    subroutine m_init( this, X, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: y

        class(BaseActivationFunction), pointer :: pt_act_fun
        type(ActivationFunctionList),  pointer :: pt_act_fun_list
        integer :: X_shape(2), Y_shape(2) 
        integer :: i
        
        if( .not. this % is_init ) then
        
            !* 从文件读取参数信息
            call this % load_NNParameter()          
            call this % allocate_memory()
            call this % load_NNParameter_array()
            call this % load_NNActivation_Function_List()
            
            X_shape = SHAPE(X)
            Y_shape = SHAPE(y)
            
            !* 输入层结点数目
            this % layers_node_count(0) = X_shape(1)
            !* 输出层结点数目
            this % layers_node_count(this % layers_count) =  &
                Y_shape(1)
            !* 样本数目
            this % sample_count = X_shape(2)         
            
        
            !* 复制训练的数据与目标，
            !*     注：不用指针是因为在当前类中可能改变X, y
            allocate( this % X, SOURCE=X )
            allocate( this % y, SOURCE=y )
            
            
            !* 初始化 my_NNStructure
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

    !* 训练函数
    subroutine m_train( this, X, t, y )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* X 是输入值，t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: X
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(inout) :: y
        
        integer :: sample_index, t_step
        real(PRECISION) :: err
        character(len=180) :: msg
          
        !* 初始化
        call this % init( X, t )
        
        do t_step=1, this % train_step
        
            call LogDebug("NNTrain: SUBROUTINE m_train step")
            
            do sample_index=1, this % sample_count

                call this % my_NNStructure % backward_propagation( X(:, sample_index), &
                    t(:, sample_index), y(:, sample_index) )
        
                !* 标准BP算法在此处更新网络权值和阈值
                if (TRIM(ADJUSTL(this % bp_algorithm)) == 'standard') then
                    call this % standard_BP_update()
                end if
            end do
            
            !* 累积BP算法在此处更新网络权值和阈值 
            if (TRIM(ADJUSTL(this % bp_algorithm)) == 'accumulation') then
                    call this % accumulation_BP_update()
            end if
            
            call this % get_total_error(t, y, err)
            
            !* undo：输出信息，便于知道进展
            !* write(*,*) "xxxx"
            !* call LogInfo("xxxx")
            write(*, *) "t_step = ", t_step, ",  err = ", err

            if (err < this % error_avg) then
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
        
        if( .not. this % is_init ) then
            call LogErr("NNTrain: SUBROUTINE m_sim, &
                NNTrain need init first.")
        end if
        
        do sample_index=1, this % sample_count
            call this % my_NNStructure % forward_propagation( X(:, sample_index), &
                t(:, sample_index), y(:, sample_index) )
        end do
        
        !* undo：根据已有的 t，预测的 y，计算误差等等.
        !* call this % get_error(t, y)
        
        return
    end subroutine m_sim
    !====
      
    !* 误差计算等
    subroutine m_get_total_error( this, t, y, err )
    implicit none
        class(NNTrain), intent(inout) :: this
        !* t 是实际输出，y 是网络预测输出
        real(PRECISION), dimension(:,:), intent(in) :: t
        real(PRECISION), dimension(:,:), intent(in) :: y
        real(PRECISION), intent(inout) :: err
        
        integer :: t_shape(2)        
        
        t_shape = SHAPE(t)
        
        err = SUM(ABS(t - y))
        err = err / t_shape(2)
        
        call LogDebug("NNTrain: SUBROUTINE m_get_total_error")
             
        return
    end subroutine m_get_total_error
    !====

    !* 标准BP算法.
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
                
            !* W = W - η * dW
            W = W - eta_w * dW
            
            !* θ = θ - η * dTheta
            Theta = Theta -eta_theta * dTheta
           
            end associate
        end do
        
        return
    end subroutine m_standard_BP_update
    !====
    
    
    !* 累积BP算法.
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
                      
            !* W = W - η * ∑ dW
            W = W - eta_w * sum_dW

            !* θ = θ - η * ∑ dTheta
            Theta = Theta - eta_theta * sum_dTheta
            
            !* 每结束一轮必须清 0
            sum_dW = 0
            sum_dTheta = 0
                
            end associate
        end do
        
        return
    end subroutine m_accumulation_BP_update
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
        
        !* 读取权值、阈值学习速率数组
        open( UNIT=30, FILE=this % NNLearningRate_file, &
            form='formatted', status='old' )            
        read( 30, * ) this % learning_rate_weight 
        read( 30, * ) this % learning_rate_threshold
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
        
        l_count = this % layers_count
        
        !* 读取每个隐藏层的结点数目
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
    
    !* 申请内存空间
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
    

    !* 销毁内存空间
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