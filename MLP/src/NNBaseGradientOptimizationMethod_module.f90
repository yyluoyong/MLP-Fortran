module mod_BaseGradientOptimizationMethod
use mod_NNStructure
implicit none
    
!-----------------------
! �����ࣺ�ݶ��Ż��㷨 |
!-----------------------
type, abstract, public :: BaseGradientOptimizationMethod

!||||||||||||    
contains   !|
!||||||||||||

    !* ��������ṹ
    procedure(abs_set_NN), deferred, public :: set_NN
	
	!* ���õ�����ʱ�䲽����Ϊѧϰ�ʿ�����ʱ�����
	procedure(abs_set_iterative_step), deferred, public :: set_iterative_step
	
	!* ����������Ĳ���
	procedure(abs_update_NN), deferred, public :: update_NN
	
	!* ǰ������
	procedure(abs_pre_process), deferred, public :: pre_process
	
	!* ������
	procedure(abs_post_process), deferred, public :: post_process
   

end type BaseGradientOptimizationMethod
!===================
    

!-------------------
! �����ࣺ�����ӿ� |
!-------------------	
abstract interface   

	!* ��������ṹ
	subroutine abs_set_NN( this, nn_structrue )
    import :: BaseGradientOptimizationMethod
    import :: NNStructure
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this
		class(NNStructure), target, intent(in) :: nn_structrue

	end subroutine
	!====
	
	!* ����������Ĳ���
	subroutine abs_update_NN( this )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this

	end subroutine
	!====
	
	!* ���õ�����ʱ�䲽
	subroutine abs_set_iterative_step( this, step )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this
		integer, intent(in) :: step

	end subroutine
	!====
	
	!* ǰ������
	subroutine abs_pre_process( this )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this

	end subroutine
	!====
	
	!* ������
	subroutine abs_post_process( this )
    import :: BaseGradientOptimizationMethod
	implicit none
		class(BaseGradientOptimizationMethod), intent(inout) :: this

	end subroutine
	!====

end interface
!===================
    
end module