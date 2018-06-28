!* 该模块定义了神经网络权值和阈值的初始化方法。
module mod_NNWeightThresholdInitMethods
use mod_Log
use mod_Precision
use mod_NNStructure
implicit none

    public :: NN_weight_threshold_init_main

    private :: m_zero_init
    private :: m_one_init
    private :: m_xavier_init
    private :: m_xavier_init_weight
    private :: m_weight_range_custom
    private :: m_threshold_range_custom
    
    contains

    !* 处理神经网络权重和阈值初始化过程
    subroutine NN_weight_threshold_init_main( &
        init_method_name, my_NNStructure )   
    implicit none
        character(len=*), intent(in) :: init_method_name
        class(NNStructure), intent(inout) :: my_NNStructure
    
        logical :: init_status
        
        call my_NNStructure % get_init_basic_status(init_status)
        
        if (init_status == .FALSE.) then
            call LogErr("mod_NNWeightThresholdInitMethods: &
                SUBROUTINE NN_weight_whreshold_init_main," ) 
            call LogErr("my_NNStructure didn't init.")
            stop
        end if
        
        
		call LogInfo("mod_NNWeightThresholdInitMethods: &
			SUBROUTINE NN_weight_whreshold_init_main")
		
        !* Append：增加初始化方法
        select case (TRIM(ADJUSTL(init_method_name)))
        case ('')
            call LogInfo("--> my_NNStructure init default.")
        case('zero')
            call m_zero_init(my_NNStructure)
            call LogInfo("--> my_NNStructure init zero.")
        case('one')
            call m_one_init(my_NNStructure)
            call LogInfo("--> my_NNStructure init [0,1].")
        case ('xavier')
            call m_xavier_init(my_NNStructure)
            call LogInfo("--> my_NNStructure init xavier.")
        case default
            call LogErr("mod_NNWeightThresholdInitMethods:   &
                SUBROUTINE NN_weight_whreshold_init_main")
            call LogErr("--> my_NNStructure didn't init.")
            stop       
        end select
        
        return
        end subroutine
    !====
    
    !* 初始化权重和阈值为 0
    subroutine m_zero_init( my_NNStructure )
    implicit none
        class(NNStructure), intent(inout) :: my_NNStructure
    
        integer :: l_count
        integer :: layer_index
        
        l_count = my_NNStructure % layers_count
        
        do layer_index=1, l_count     
            associate (                                                     &
                W      => my_NNStructure % pt_W( layer_index ) % W,         &
                Theta  => my_NNStructure % pt_Theta( layer_index ) % Theta  &
            )
            
            Theta = 0
            W = 0
                
            end associate
        end do
        
        call LogInfo("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_zero_init.")
        
        return
    end subroutine m_zero_init
    !====  
    
    !* 权值、阈值初始化(0,1)
    subroutine m_one_init( my_NNStructure )
    implicit none
        class(NNStructure), intent(inout) :: my_NNStructure
    
        call m_weight_range_custom( my_NNStructure, 0.0, 1.0 )
        call m_threshold_range_custom( my_NNStructure, 0.0, 1.0 )
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_one_init.")
        
        return
    end subroutine m_one_init
    !====
    
    !* 权重初始化在指定的范围
    subroutine m_weight_range_custom( my_NNStructure, &
        set_min, set_max )
    implicit none
        class(NNStructure), intent(inout) :: my_NNStructure
        real(PRECISION) :: set_min, set_max
        
        integer :: l_count
        integer :: layer_index
        
        l_count = my_NNStructure % layers_count
        
        do layer_index=1, l_count     
            associate (                                        &
                W => my_NNStructure % pt_W( layer_index ) % W  &
            )
            
            W = (set_max - set_min) * W + set_min
                
            end associate
        end do
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_weight_range_custom.")
        
        return
    end subroutine m_weight_range_custom
    !====
    
    
    !* 阈值初始化在指定的范围
    subroutine m_threshold_range_custom( my_NNStructure, &
        set_min, set_max )
    implicit none
        class(NNStructure), intent(inout) :: my_NNStructure
        real(PRECISION) :: set_min, set_max
        
        integer :: l_count
        integer :: layer_index
        
        l_count = my_NNStructure % layers_count
        
        do layer_index=1, l_count     
            associate (                                                    &
                Theta => my_NNStructure % pt_Theta( layer_index ) % Theta  &
            )
            
            Theta = (set_max - set_min) * Theta + set_min
                
            end associate
        end do
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_threshold_range_custom.")
        
        return
    end subroutine m_threshold_range_custom
    !====

    
    !* Xavier 初始化权重和阈值
    subroutine m_xavier_init( my_NNStructure )
    implicit none
        class(NNStructure), intent(inout) :: my_NNStructure
    
        integer :: l_count, M, N
        integer :: layer_index
        
        l_count = my_NNStructure % layers_count
        
        do layer_index=1, l_count     
            associate (                                                     &
                W      => my_NNStructure % pt_W( layer_index ) % W,         &
                Theta  => my_NNStructure % pt_Theta( layer_index ) % Theta  &
            )
        
            M = my_NNStructure % layers_node_count(layer_index - 1)
			N = my_NNStructure % layers_node_count(layer_index)
            
            Theta = 0
            call m_xavier_init_weight(M, N, W)
                
            end associate
        end do
        
        call LogDebug("mod_NNWeightThresholdInitMethods: &
            SUBROUTINE m_xavier_init.")
        
        return
    end subroutine m_xavier_init
    !====    
        
        
    !* Xavier 初始化权重
    !* W ~ U(-x, x) 的均匀分布，x = sqrt[6 / (n_j + n_{j+1})]
    !* n_j，n_{j+1} 是 W 连接两层的结点数.
    !* From paper: Xavier Glorot, Yoshua Bengio 
    !*   Understanding the difficulty of training deep feed.
    subroutine m_xavier_init_weight( count_layer_node_present, &
        count_layer_node_out, weight )
    implicit none
        integer, intent(in) :: count_layer_node_present, &
            count_layer_node_out
        real(PRECISION), dimension(:,:), intent(inout) :: weight
    
        real(PRECISION) :: x
        integer :: M, N
        
        M = count_layer_node_present
        N = count_layer_node_out
        
        x = SQRT(6.0 / (M + N))
        
        call RANDOM_SEED()
        call RANDOM_NUMBER(weight)
        
        weight = 2.0 * x * weight - x
        
        return
    end subroutine m_xavier_init_weight
    !====
    
    
end module