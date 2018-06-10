module mod_ActivationFunctionList
use mod_BaseActivationFunction
use mod_Sigmod
use mod_TanH
use mod_ReLU
use mod_PReLU
use mod_ELU
use mod_Linear
use mod_Log
implicit none

!----------------------
! 工作类：激活函数列表 |
!----------------------
type, public :: ActivationFunctionList
    !* 注：增加激活函数，应修改 Append 标识处的代码.
    
    !* 是否初始化完成的标识
    logical, private :: is_init = .false.
    
   
    !!* 实现的激活函数数量
    !!* Append：增加激活函数
    !integer, public :: ACTIVATION_FUNCTION_COUNT = 6  
    
    !* Append：增加激活函数
    type(Sigmod), pointer, private :: function_sigmod
    type(Tan_H),  pointer, private :: function_tanh
    type(ReLU),   pointer, private :: function_ReLU
    type(PReLU),  pointer, private :: function_PReLU
    type(ELU),    pointer, private :: function_ELU
    type(Linear), pointer, private :: function_linear


!||||||||||||    
contains   !|
!||||||||||||

    procedure, private :: init => m_init
    
    procedure, public :: get_activation_function_by_name => m_get_act_fun_by_name

end type ActivationFunctionList
!===================

    private :: m_init
    private :: m_get_act_fun_by_name

    
	
!||||||||||||    
contains   !|
!||||||||||||
    
    !* 初始化
    subroutine m_init( this )
    implicit none
        class(ActivationFunctionList), intent(inout) :: this 

        if( .not. this % is_init ) then
        
            !* Append：增加激活函数
            allocate( this % function_sigmod )
            allocate( this % function_tanh   )
            allocate( this % function_ReLU   )
            allocate( this % function_PReLU  )
            allocate( this % function_ELU    )
            allocate( this % function_linear )
        
            this % is_init = .true.    

        end if

        return
    end subroutine m_init
    !====

    !* 根据激活函数名字选取相应的激活函数
    subroutine m_get_act_fun_by_name( this, act_fun_name, pt_act_fun )
    implicit none
        class(ActivationFunctionList), intent(inout) :: this 
        character(len=*), intent(in) :: act_fun_name
        class(BaseActivationFunction), pointer, intent(out) :: pt_act_fun
        
        call this % init()
        
        !* Append：增加激活函数
        select case (TRIM(ADJUSTL(act_fun_name)))
        case ('sigmod')
            pt_act_fun => this % function_sigmod
        case ('tanh')
            pt_act_fun => this % function_tanh
        case ('ReLU')
            pt_act_fun => this % function_ReLU
        case ('PReLU')
            pt_act_fun => this % function_PReLU
        case ('ELU')
            pt_act_fun => this % function_ELU
        case ('linear')
            pt_act_fun => this % function_linear
        case default
            call LogErr("ActivationFunctionList: SUBROUTINE m_get_act_fun_by_name, &
                    act_fun_index > act_fun_count, activation function name Error.")
            stop       
        end select
        
        call LogInfo("ActivationFunctionList: SUBROUTINE m_get_act_fun_by_name, &
            act_fun_name is : ")
        call LogInfo(TRIM(ADJUSTL(act_fun_name)))
    
        return
    end subroutine m_get_act_fun_by_name
    !====
    

    
end module